# Mac 준비

[START-HERE](../START-HERE.md)대로 포크해서 받은 저장소 폴더를 에이전트에서 엽니다. 이 문서의 명령은 모두 그 폴더를 현재 작업 폴더로 삼습니다. 설치된 도구를 먼저 확인하고, 빠진 것만 준비합니다.

## 1. 현재 상태 확인

지원·테스트하는 Python은 **3.12와 3.13**이고, 처음 설치한다면 **3.13**을 권장합니다. 이미 있다면 추가 패키지 없이 바로 검사할 수 있습니다.

```sh
python3.13 -B scripts/setup_environment.py --check
```

빠진 항목이 있으면 종료 코드 `1`과 함께 항목별 경로와 오류를 출력합니다. 이 검사는 아무것도 설치하거나 바꾸지 않습니다.

## 2. Python · FFmpeg · Node.js

[Homebrew](https://brew.sh/)는 공통강의 Git 세팅에서 설치했습니다. 없다면 공식 설치 명령을 **학생이 터미널 앱에서 직접** 실행합니다. 관리자 암호를 입력해야 해서 에이전트가 대신 끝낼 수 없습니다.

```sh
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

이미 있는 항목은 아래 목록에서 뺍니다. 각 공식 안내는 [Python 3.13](https://formulae.brew.sh/formula/python@3.13), [FFmpeg](https://formulae.brew.sh/formula/ffmpeg), [Node.js](https://formulae.brew.sh/formula/node)입니다. COLMAP 앱은 필요하지 않습니다. 다음 단계에서 pycolmap이 함께 설치됩니다.

```sh
TASK_BREW="$(command -v brew || true)"
[ -x "$TASK_BREW" ] || TASK_BREW=/opt/homebrew/bin/brew
"$TASK_BREW" install python@3.13 ffmpeg node
TASK_PYTHON="$("$TASK_BREW" --prefix python@3.13)/bin/python3.13"
```

## 3. 이 폴더에만 설치

`--install`은 이 폴더 안에만 설치합니다. `.venv`를 만들고 `requirements.txt`(numpy, OpenCV, SciPy, pycolmap, ModernGL)를 설치한 뒤, `npm ci --prefix Source`로 three.js를 받습니다. 시스템 Python이나 전역 PATH는 바꾸지 않습니다. 다른 컴퓨터에서 복사해 온 `.venv`는 쓰지 말고 이 컴퓨터에서 새로 만듭니다.

```sh
"$TASK_PYTHON" -B scripts/setup_environment.py --install
.venv/bin/python -B scripts/setup_environment.py --configure
.venv/bin/python -B scripts/setup_environment.py --check
```

`--configure`는 찾은 `ffmpeg`·`ffprobe`·`node`·`npm`의 절대 경로를 `tools.local.json`에 저장합니다. 이 파일은 Git에 올라가지 않습니다. 도구를 찾는 순서는 **명령 인자 → 환경변수(`FFMPEG_BINARY` 등) → tools.local.json → PATH**입니다.

`--check`에서 `"ready": true`가 나오고 `opengl` 항목에 `GL_VERSION`이 보이면 준비가 끝난 것입니다. Apple Silicon Mac은 `4.1 Metal`로 표시됩니다. 이후 문서에서 `python`은 `.venv/bin/python`을 뜻합니다.

## 4. 첫 실행

```sh
.venv/bin/python -B -m pipeline composite hall_water
```

외장 SSD를 쓰려면 `--work /Volumes/드라이브/폴더`를 붙입니다. 이어지는 단계는 [워크플로](workflow.md), 긴 계산을 맡기는 법은 [background](background.md)에 있습니다.

## 5. 다른 예제를 쓸 때

`--check` 출력의 `examples` 항목이 예제마다 빠진 것을 알려 줍니다.

| 예제 | 준비 |
| --- | --- |
| Shader Live | `.venv/bin/python scripts/fetch_assets.py footage 10708812` |
| 무중력 과일 | `.venv/bin/python -B scripts/setup_environment.py --install --with body`, `scripts/fetch_assets.py footage 7525471`, `scripts/fetch_assets.py pose-model` |

이 예제들은 `ffmpeg`와 `ffprobe`를 PATH에서 찾습니다. Homebrew로 설치했다면 바로 잡힙니다. 이 예제들은 Windows에서만 확인했고 Mac에서는 실행해 보지 못했습니다.
