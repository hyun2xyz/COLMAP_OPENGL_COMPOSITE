# 계산은 Python, 다음 확인은 일회성 예약

카메라 계산이나 렌더가 길어진다고 매초 LLM 토큰이 쓰이는 것은 아닙니다.
이 구조는 긴 로그를 문맥에 넣거나 에이전트를 반복 호출하는 일을 줄이고,
응답이 끝난 동안에도 실제 로컬 계산이 계속되도록 합니다.
Python 계산기와 LLM을 다시 깨우는 예약기는 서로 다른 기능입니다.

이 패키지에서 4초 숏 하나의 `composite`는 M2 Max에서 1분 안팎입니다. 느린 컴퓨터, 긴 숏, 여러 숏을 이어 돌릴 때만 분리 실행을 씁니다. 저장소 폴더에서 이렇게 시작합니다(Windows는 `.\.venv\Scripts\python.exe`).

```sh
python scripts/jobs.py start --job "$PWD/work/jobs/hall-composite-1" --estimate-minutes 3 --timeout-minutes 30 -- .venv/bin/python -B -m pipeline composite hall_water
python scripts/jobs.py status --job "$PWD/work/jobs/hall-composite-1"
```

## 시작할 때

1. `jobs.py start`로 새 절대 job 폴더를 만듭니다. supervisor/계산 프로세스가
   부모 CLI와 분리되고 stdout/stderr는 파일에 기록됩니다.
2. 작업 ID·예상 시간 **범위와 근거**·다음 확인 시각을 안내합니다.
   첫 실행은 ETA 미확인으로 표시합니다. `--estimate-minutes`는 사용자가
   정한 거친 추정치이며, 모델이 실제 남은 시간을 측정했다는 뜻이 아닙니다.
3. `work/<숏>/next.md`에 job 경로·기대 결과·다음 단계·확인 횟수를 남깁니다.
4. 현재 채팅을 대상으로 한 번만 재개할 예약을 실제 도구로 생성합니다.
   생성 성공·ID·다음 시각을 확인한 뒤 아래처럼 **기록만** 합니다.

```sh
python scripts/jobs.py wakeup --job /absolute/job --automation-id REAL_ID --at 2026-10-25T14:10:00+09:00
```

시각은 예시이며 실제 현재 날짜·시간대에 맞춰야 합니다. 이 명령 자체는
예약을 만들지 않습니다. `record_only`/`external_automation_changed:false`가 출력됩니다.
예약 기능이 없거나 생성이 실패하면 그 사실과 상태 확인 명령을 알려 줍니다.
독립적으로 할 일이 남지 않으면 현재 응답을 끝냅니다. LLM 대기 루프는 만들지 않습니다.

## 다시 깨어났을 때

예약 프롬프트에는 **job ID/절대 job 경로/스킬 경로/다음 한 단계/최대 시간과
확인 횟수/자기 예약 정리**를 포함합니다. 예시:

> 지정된 job의 계산 완료만 한 번 확인한다. 먼저 이 실행의 예약 ID를
> 삭제하거나 자동 삭제되었음을 예약 도구로 확인한다. 다른 예약은 건드리지
> 않는다. status를 한 번 읽고 정상 종료와 결과를 검증한다. 진행 중이면
> 기존 계산은 유지하고 다음 확인 예약 하나만 만든 뒤 응답을 끝낸다.
> 실패/정체/확인 한도 초과에는 재실행하지 않고 보고한다.

외부 예약이 사라진 것을 확인한 후 같은 ID의 로컬 기록만 정리합니다.

```sh
python scripts/jobs.py clear-wakeup --job /absolute/job --automation-id REAL_ID
python scripts/jobs.py status --job /absolute/job
```

기본 status는 로그를 출력하지 않습니다. 실패 원인이 필요할 때만 `--tail 10`처럼
제한해서 읽고, 긴 로그에서는 해당 에러 구간만 추립니다.

| 상태 | 다음 행동 |
| --- | --- |
| `starting/running` | 중복 계산 없이 한 번만 다음 예약. 진행/남은 시간에 맞춰 간격 늘리기 |
| `succeeded` | 종료 코드 0과 산출물·등록 비율을 함께 확인 후 다음 단계 |
| `failed/timeout` | 자기 예약 정리, 원인·로그 경로 보고. 새 run은 원인 수정 후 명시적으로 |
| `missing/stale/invalid` | 완료 여부 미확인. 상태/컴퓨터를 확인하고 성공이나 자동 재시작으로 해석하지 않기 |

heartbeat는 약 10초마다 갱신되며, 60초 이상 끊긴 실행 상태는 stale입니다.
이는 solver 진행률이 아니라 supervisor의 생존 신호입니다. 기본 한도는
12번 확인 또는 120분입니다. 한도 도달은 계산 완료/성공이 아니며 실행 시
설정한 timeout은 별도로 적용됩니다. timeout은 해당 작업의 자식 트리만 종료합니다.
취소 요청에는 예약을 정리하고, 실제 계산까지 멈출지 요청 범위대로 처리합니다.
이 입문 실행기는 재부팅 후 자동 재시작이나 자동 복구를 하지 않습니다.

## Codex와 Claude Code

**Codex desktop:** 현재 채팅의 heartbeat 예약 도구를 사용합니다. 한 번만
확인하는 예약으로 만들고, 재개 첫 단계에서 해당 ID를 삭제·확인합니다.
기본 제공 예약의 저장 항목이 저절로 사라진다고 가정하지 않습니다.
도구가 공개한 실제 스키마를 읽고 호출하며 로컬 automation 파일을 직접
고치거나 반복 cron을 몰래 설치하지 않습니다. Mac/Windows에서 같은 원칙입니다.
[공식 안내](https://learn.chatgpt.com/docs/automations#schedule-a-task-inside-a-chat).

**Claude Code:** 현재 세션에 `CronCreate/CronList/CronDelete`가 있으면
자연어로 요청한 **one-shot**을 사용합니다. 공식 문서상 한 번 실행된 작업은
스스로 삭제됩니다. 재개 시 ID가 사라졌는지 확인하고 남아 있으면 그 ID만 정리합니다.
고정 반복 `/loop`나 기본 유지보수 루프로 바꾸지 않습니다. 세션을 열어 두어야
하며 닫힌 동안 지나간 일회성 예약이 복원된다고 보장하지 않습니다.
[공식 안내](https://code.claude.com/docs/en/scheduled-tasks#set-a-one-time-reminder).

에이전트/계정/버전에 예약 도구가 없으면 계산은 동일하게 분리 실행할 수 있지만
자동 채팅 재개는 보장할 수 없습니다. 상태 명령과 `이 job 경로에서 이어 주세요`라는
재개 문장을 남깁니다. 예약 성공을 꾸미거나 사용자 토큰/비공식 브리지를 쓰지 않습니다.

로컬 계산과 예약을 위해 컴퓨터를 켜 두고 절전으로 멈추지 않게 합니다.
Codex 앱/Claude 세션도 재개할 때 살아 있어야 합니다. 필요하다면 Mac의
`caffeinate` 같은 OS 기능을 학생에게 설명한 후 선택적으로 사용합니다.
