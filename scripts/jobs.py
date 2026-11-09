#!/usr/bin/env python3
"""Small, dependency-free detached job runner for classroom commands.

Each job directory is single-use. ``start`` launches a detached supervisor, which
owns the command process and writes status.json atomically. No command is ever
resumed from saved state. Wakeup commands only record references to automations
created or deleted separately through the assistant's automation tool.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
import uuid


POLL_SECONDS = 0.5
HEARTBEAT_SECONDS = 10.0
STALE_SECONDS = 60.0
TERMINATE_GRACE_SECONDS = 2.0
MAX_TAIL_LINES = 200
MAX_TAIL_BYTES = 32 * 1024
LIVE_STATES = {"starting", "running"}
TERMINAL_STATES = {"succeeded", "failed", "timeout"}


class JobError(Exception):
    """An actionable CLI error that does not need a traceback."""


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def _parse_time(value: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError("timestamp must be an ISO string with timezone")
    parsed = datetime.fromisoformat(value[:-1] + "+00:00" if value.endswith("Z") else value)
    if parsed.tzinfo is None:
        raise ValueError("timestamp must include a timezone, for example +09:00 or Z")
    return parsed.astimezone(timezone.utc)


def _absolute_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("--job must be an absolute directory path")
    return path.resolve()


def _positive_minutes(value: str) -> float:
    try:
        minutes = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("minutes must be a positive number") from exc
    if not math.isfinite(minutes) or minutes <= 0:
        raise argparse.ArgumentTypeError("minutes must be a finite positive number")
    return minutes


def _tail_count(value: str) -> int:
    try:
        count = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--tail must be a whole number") from exc
    if not 0 <= count <= MAX_TAIL_LINES:
        raise argparse.ArgumentTypeError(f"--tail must be between 0 and {MAX_TAIL_LINES}")
    return count


def _print_json(value: dict, *, stream=None) -> None:
    print(json.dumps(value, ensure_ascii=False, separators=(",", ":")), file=stream)


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _read_json(path: Path) -> dict:
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, ValueError) as exc:
        raise JobError(f"cannot read {path.name}: {exc}") from exc
    if not isinstance(value, dict):
        raise JobError(f"{path.name} must contain a JSON object")
    return value


def _manifest(job: Path) -> dict:
    value = _read_json(job / "job.json")
    command = value.get("command")
    if (
        value.get("version") != 1
        or not isinstance(value.get("job_id"), str)
        or not value["job_id"]
        or not isinstance(command, list)
        or not command
        or not all(isinstance(argument, str) for argument in command)
        or not isinstance(value.get("cwd"), str)
    ):
        raise JobError("job.json is not a valid job manifest")
    for field in ("estimate_seconds", "timeout_seconds"):
        number = value.get(field)
        if isinstance(number, bool) or not isinstance(number, (int, float)):
            raise JobError(f"job.json has an invalid {field}")
        if not math.isfinite(number) or number <= 0:
            raise JobError(f"job.json has an invalid {field}")
    return value


def _logs(job: Path) -> dict:
    return {name: str(job / f"{name}.log") for name in ("stdout", "stderr", "runner")}


def _tail(path: Path, count: int) -> str:
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - MAX_TAIL_BYTES))
            data = handle.read(MAX_TAIL_BYTES)
    except FileNotFoundError:
        return ""
    return "\n".join(data.decode("utf-8", errors="replace").splitlines()[-count:])


def _current_wakeup(job: Path, job_id: str) -> dict | None:
    if not (job / "wakeup.json").exists():
        return None
    try:
        record = _read_json(job / "wakeup.json")
        if record.get("job_id") != job_id or not isinstance(record.get("automation_id"), str):
            raise JobError("wakeup.json does not match this job")
        _parse_time(record["at"])
        return {"automation_id": record["automation_id"], "nextcheck_at": record["at"]}
    except (JobError, KeyError, TypeError, ValueError, OverflowError) as exc:
        return {"state": "invalid", "error": str(exc)}


def _status(job: Path, tail: int = 0) -> dict:
    summary = {
        "job_id": None,
        "state": "missing",
        "pid": None,
        "worker_pid": None,
        "command": [],
        "started_at": None,
        "finished_at": None,
        "elapsed_seconds": None,
        "heartbeat_at": None,
        "heartbeat_recent": False,
        "exit_code": None,
        "logs": _logs(job),
        "estimated_finish_at": None,
    }
    if not job.is_dir():
        summary["error"] = "job directory does not exist"
        return summary
    try:
        manifest = _manifest(job)
        summary["job_id"] = manifest["job_id"]
        summary["command"] = manifest["command"]
        wakeup = _current_wakeup(job, manifest["job_id"])
        if wakeup is not None:
            summary["wakeup"] = wakeup
        if not (job / "status.json").exists():
            summary["error"] = "status.json is missing; completion is unknown"
            return summary
        record = _read_json(job / "status.json")
        state = record.get("state")
        if record.get("job_id") != manifest["job_id"] or state not in LIVE_STATES | TERMINAL_STATES:
            raise JobError("status.json does not match the job or has an invalid state")
        started = _parse_time(record["started_at"])
        heartbeat = _parse_time(record["heartbeat_at"])
        if state in TERMINAL_STATES:
            _parse_time(record["finished_at"])
        exit_code = record.get("exit_code")
        if exit_code is not None and (isinstance(exit_code, bool) or not isinstance(exit_code, int)):
            raise JobError("status.json has an invalid exit_code")
        if state == "succeeded" and exit_code != 0:
            raise JobError("a succeeded status must have exit_code 0")
        now = _now()
        heartbeat_age = (now - heartbeat).total_seconds()
        recent = -STALE_SECONDS <= heartbeat_age <= STALE_SECONDS
        elapsed = record.get("elapsed_seconds", 0.0)
        if isinstance(elapsed, bool) or not isinstance(elapsed, (float, int)) or not math.isfinite(elapsed):
            raise JobError("status.json has an invalid elapsed_seconds")
        if state in LIVE_STATES:
            elapsed = max(0.0, (now - started).total_seconds())
            if not recent:
                state = "stale"
        summary.update(
            state=state,
            pid=record.get("pid"),
            worker_pid=record.get("worker_pid"),
            started_at=record["started_at"],
            finished_at=record.get("finished_at"),
            elapsed_seconds=round(max(0.0, elapsed), 3),
            heartbeat_at=record["heartbeat_at"],
            heartbeat_recent=recent,
            exit_code=exit_code,
            estimated_finish_at=record.get("estimated_finish_at"),
        )
        if "error" in record:
            summary["error"] = record["error"]
        if state == "stale":
            summary["error"] = "heartbeat is stale; command completion is unknown; no automatic restart"
    except (JobError, KeyError, TypeError, ValueError, OverflowError) as exc:
        summary.update(state="invalid", error=str(exc))
    if tail:
        summary["log_tail"] = {
            name: _tail(job / f"{name}.log", tail) for name in ("stdout", "stderr")
        }
    return summary


def _status_exit_code(summary: dict) -> int:
    return 0 if summary["state"] in LIVE_STATES | {"succeeded"} else 1


def _detach_options() -> dict:
    if os.name == "nt":
        return {
            "creationflags": subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
        }
    return {"start_new_session": True}


def _start(args: argparse.Namespace) -> int:
    command = args.command[1:] if args.command and args.command[0] == "--" else args.command
    if not command:
        raise JobError("a command is required after --")
    if any("\0" in argument for argument in command):
        raise JobError("command arguments cannot contain NUL characters")
    try:
        args.job.mkdir(mode=0o700, parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise JobError("job directory already exists; use a new directory for every run") from exc
    created = _now()
    manifest = {
        "version": 1,
        "job_id": uuid.uuid4().hex,
        "command": command,
        "cwd": str(Path.cwd()),
        "created_at": _iso(created),
        "estimate_seconds": args.estimate_minutes * 60.0,
        "timeout_seconds": args.timeout_minutes * 60.0,
    }
    record = {
        "job_id": manifest["job_id"],
        "state": "starting",
        "pid": None,
        "worker_pid": None,
        "started_at": _iso(created),
        "finished_at": None,
        "heartbeat_at": _iso(created),
        "elapsed_seconds": 0.0,
        "exit_code": None,
        "estimated_finish_at": _iso(created + timedelta(seconds=manifest["estimate_seconds"])),
    }
    _atomic_json(args.job / "job.json", manifest)
    _atomic_json(args.job / "status.json", record)
    try:
        with (args.job / "runner.log").open("ab", buffering=0) as runner_log:
            worker = subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()), "_worker", "--job", str(args.job)],
                stdin=subprocess.DEVNULL,
                stdout=runner_log,
                stderr=subprocess.STDOUT,
                close_fds=True,
                shell=False,
                **_detach_options(),
            )
    except (OSError, ValueError) as exc:
        record.update(
            state="failed",
            finished_at=_iso(_now()),
            heartbeat_at=_iso(_now()),
            error=f"could not start supervisor: {exc}",
        )
        _atomic_json(args.job / "status.json", record)
        _print_json(_status(args.job))
        return 1
    summary = _status(args.job)
    # Do not race the supervisor by overwriting status.json with its PID.
    if summary["worker_pid"] is None:
        summary["worker_pid"] = worker.pid
    _print_json(summary)
    return _status_exit_code(summary)


def _terminate_owned_child(process: subprocess.Popen) -> str | None:
    """Terminate this live child's POSIX group or Windows process tree."""
    if process.poll() is not None:
        return
    warning = None
    if os.name == "nt":
        taskkill = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "taskkill.exe"
        try:
            result = subprocess.run(
                [str(taskkill), "/PID", str(process.pid), "/T", "/F"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
                shell=False,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            if result.returncode:
                warning = f"Windows tree termination returned {result.returncode}; inspect descendants"
        except (OSError, subprocess.TimeoutExpired) as exc:
            warning = f"Windows tree termination failed: {exc}; inspect descendants"
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        elif process.poll() is None:
            process.terminate()
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=TERMINATE_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        pass
    # The direct child may exit on TERM while its grandchild keeps running.
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        elif process.poll() is None:
            process.kill()
    except ProcessLookupError:
        pass
    process.wait()
    return warning


def _interrupted(signum, _frame) -> None:
    raise JobError(f"supervisor received signal {signum}")


def _worker(job: Path) -> int:
    # Kept permanently: a supervisor must never rerun a completed or stale job.
    try:
        lock = os.open(job / "worker.lock", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        return 1
    with os.fdopen(lock, "w", encoding="ascii") as handle:
        handle.write(str(os.getpid()) + "\n")
    manifest = _manifest(job)
    record = _read_json(job / "status.json")
    if record.get("job_id") != manifest["job_id"] or record.get("state") != "starting":
        return 1
    record["worker_pid"] = os.getpid()
    process = None
    started_monotonic = time.monotonic()
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, _interrupted)
    try:
        started = _now()
        record.update(
            started_at=_iso(started),
            heartbeat_at=_iso(started),
            estimated_finish_at=_iso(started + timedelta(seconds=manifest["estimate_seconds"])),
        )
        _atomic_json(job / "status.json", record)
        with (job / "stdout.log").open("ab", buffering=0) as stdout_log, (
            job / "stderr.log"
        ).open("ab", buffering=0) as stderr_log:
            process = subprocess.Popen(
                manifest["command"],
                cwd=manifest["cwd"],
                stdin=subprocess.DEVNULL,
                stdout=stdout_log,
                stderr=stderr_log,
                close_fds=True,
                shell=False,
                **_detach_options(),
            )
            record.update(state="running", pid=process.pid, heartbeat_at=_iso(_now()))
            _atomic_json(job / "status.json", record)
            last_heartbeat = time.monotonic()
            while True:
                elapsed = time.monotonic() - started_monotonic
                if time.monotonic() - last_heartbeat >= HEARTBEAT_SECONDS:
                    record.update(heartbeat_at=_iso(_now()), elapsed_seconds=round(elapsed, 3))
                    _atomic_json(job / "status.json", record)
                    last_heartbeat = time.monotonic()
                remaining = manifest["timeout_seconds"] - elapsed
                if process.poll() is not None:
                    record.update(
                        state="succeeded" if process.returncode == 0 else "failed",
                        exit_code=process.returncode,
                    )
                    break
                if remaining <= 0:
                    # Preserve timeout even if the command exits 0 while handling SIGTERM.
                    record.update(state="timeout", error="command exceeded timeout")
                    warning = _terminate_owned_child(process)
                    if warning:
                        record["error"] += "; " + warning
                    record["exit_code"] = process.returncode
                    break
                try:
                    process.wait(timeout=min(POLL_SECONDS, remaining))
                except subprocess.TimeoutExpired:
                    pass
    except Exception as exc:
        traceback.print_exc()
        warning = None
        if process is not None:
            warning = _terminate_owned_child(process)
            record["exit_code"] = process.returncode
        if record.get("state") != "timeout":
            record["state"] = "failed"
        record["error"] = str(exc) + ("; " + warning if warning else "")
    finally:
        finished = _now()
        record.update(
            finished_at=_iso(finished),
            heartbeat_at=_iso(finished),
            elapsed_seconds=round(time.monotonic() - started_monotonic, 3),
        )
        _atomic_json(job / "status.json", record)
    return 0 if record["state"] == "succeeded" else 1


@contextmanager
def _wakeup_lock(job: Path):
    path = job / ".wakeup.lock"
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise JobError("wakeup record is locked; inspect a possibly interrupted update") from exc
    try:
        os.close(descriptor)
        yield
    finally:
        path.unlink()


def _automation_id(value: str) -> str:
    if not value or value != value.strip() or len(value) > 256:
        raise argparse.ArgumentTypeError("automation ID must be a nonempty ID without surrounding spaces")
    return value


def _wakeup(args: argparse.Namespace) -> int:
    manifest = _manifest(args.job)
    try:
        at = _iso(_parse_time(args.at))
    except (ValueError, OverflowError) as exc:
        raise JobError(f"--at must be an ISO timestamp with timezone: {exc}") from exc
    path = args.job / "wakeup.json"
    with _wakeup_lock(args.job):
        if path.exists():
            existing = _read_json(path)
            if (
                existing.get("automation_id") != args.automation_id
                or existing.get("job_id") != manifest["job_id"]
            ):
                raise JobError("another automation or job ID is recorded; inspect it before updating")
        record = {
            "job_id": manifest["job_id"],
            "automation_id": args.automation_id,
            "at": at,
            "recorded_at": _iso(_now()),
            "record_only": True,
            "external_automation_changed": False,
        }
        _atomic_json(path, record)
    _print_json(record)
    return 0


def _clear_wakeup(args: argparse.Namespace) -> int:
    manifest = _manifest(args.job)
    result = {
        "job_id": manifest["job_id"],
        "automation_id": args.automation_id,
        "cleared": False,
        "record_only": True,
        "external_automation_changed": False,
    }
    path = args.job / "wakeup.json"
    with _wakeup_lock(args.job):
        if not path.exists():
            result["reason"] = "no_wakeup_record"
        else:
            existing = _read_json(path)
            if existing.get("automation_id") != args.automation_id or existing.get("job_id") != manifest["job_id"]:
                result["reason"] = "automation_id_or_job_id_mismatch"
                _print_json(result)
                return 1
            path.unlink()
            result["cleared"] = True
    _print_json(result)
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    start = commands.add_parser("start", help="start once in a new, absolute job directory")
    start.add_argument("--job", required=True, type=_absolute_path)
    start.add_argument("--estimate-minutes", required=True, type=_positive_minutes)
    start.add_argument("--timeout-minutes", required=True, type=_positive_minutes)
    start.add_argument("command", nargs=argparse.REMAINDER, help="-- executable argument ...")
    status = commands.add_parser("status", help="read compact JSON; never restart a job")
    status.add_argument("--job", required=True, type=_absolute_path)
    status.add_argument("--tail", type=_tail_count, default=0, help="last 0..200 lines per log (default 0; max 32 KiB each)")
    wakeup = commands.add_parser("wakeup", help="record an existing external automation ID only")
    wakeup.add_argument("--job", required=True, type=_absolute_path)
    wakeup.add_argument("--automation-id", required=True, type=_automation_id)
    wakeup.add_argument("--at", required=True, help="ISO timestamp including timezone")
    clear = commands.add_parser("clear-wakeup", help="remove only a matching local automation reference")
    clear.add_argument("--job", required=True, type=_absolute_path)
    clear.add_argument("--automation-id", required=True, type=_automation_id)
    worker = commands.add_parser("_worker", help=argparse.SUPPRESS)
    worker.add_argument("--job", required=True, type=_absolute_path)
    return parser


def main(argv: list[str] | None = None) -> int:
    # Captured stdout uses the Windows locale by default; the JSON CLI is UTF-8.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    args = _parser().parse_args(argv)
    try:
        if args.action == "start":
            return _start(args)
        if args.action == "status":
            summary = _status(args.job, args.tail)
            _print_json(summary)
            return _status_exit_code(summary)
        if args.action == "wakeup":
            return _wakeup(args)
        if args.action == "clear-wakeup":
            return _clear_wakeup(args)
        return _worker(args.job)
    except (JobError, OSError, OverflowError) as exc:
        _print_json({"state": "error", "error": str(exc)}, stream=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
