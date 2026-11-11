# Windows 준비

Windows 10/11 **x64**를 기준으로 합니다. GPU는 OpenGL 3.3 이상을 지원하는 드라이버면 충분합니다. 최근 몇 년 사이의 NVIDIA·AMD·Intel 그래픽 드라이버는 모두 해당합니다. 카메라 계산(COLMAP)은 CPU로 하므로 CUDA가 필요하지 않습니다. [START-HERE](../START-HERE.md)대로 포크해서 받은 저장소 폴더를 에이전트와 PowerShell의 작업 폴더로 선택합니다.

## 1. 현재 상태 확인

```powershell
py -3.13 -B scripts/setup_environment.py --check
```

빠진 항목은 종료 코드 `1`과 함께 오류를 출력하며, 이 검사는 아무것도 바꾸지 않습니다.

## 2. Python · FFmpeg · Node.js

Python은 [공식 Windows 설치기](https://www.python.org/downloads/windows/)의 3.13 x64를 쓰거나 WinGet으로 설치합니다. WinGet을 쓸 때는 [Microsoft 안내](https://learn.microsoft.com/en-us/windows/package-manager/winget/)에 따라 패키지 출처를 먼저 확인합니다.

```powershell
winget install --id Python.Python.3.13 --exact --source winget --scope user --architecture x64
winget install --id Gyan.FFmpeg --exact --source winget
winget install --id OpenJS.NodeJS.LTS --exact --source winget
```

설치한 뒤에는 **새 터미널**을 열고 `py -3.13 --version`, `ffmpeg -version`, `node --version`을 확인합니다. FFmpeg는 [공식 다운로드 페이지](https://ffmpeg.org/download.html#build-windows)가 연결하는 Gyan 빌드입니다. ZIP으로 받았다면 `bin` 폴더째로 두고 `--configure --ffmpeg ...\bin\ffmpeg.exe --ffprobe ...\bin\ffprobe.exe`로 경로를 알려 줍니다. COLMAP 앱은 필요하지 않습니다(pycolmap의 Windows 휠을 씁니다).

## 3. 이 폴더에만 설치

```powershell
py -3.13 -B scripts/setup_environment.py --install
.\.venv\Scripts\python.exe -B scripts/setup_environment.py --configure
.\.venv\Scripts\python.exe -B scripts/setup_environment.py --check
```

`--install`은 이 폴더의 `.venv`와 `Source\node_modules`만 만듭니다. 시스템 PATH와 PowerShell 실행 정책은 바꾸지 않습니다. 실행할 때 activate는 필요 없습니다. 이후 문서에서 `python`은 `.\.venv\Scripts\python.exe`를 뜻합니다.

`opengl` 항목이 오류를 내면 원격 데스크톱이나 기본 디스플레이 드라이버(Microsoft Basic Display Adapter)에서 실행 중일 수 있습니다. 그래픽 카드 제조사의 드라이버를 설치하고 실제 화면 세션에서 다시 확인합니다. 출처를 모르는 DLL은 복사하지 않습니다.

## 4. 첫 실행

```powershell
.\.venv\Scripts\python.exe -B -m pipeline composite hall_water
```

이 패키지가 Windows에서 실제로 어디까지 확인되었는지는 [검증 범위](validation.md)에 있습니다. 이어지는 단계는 [워크플로](workflow.md)를 따릅니다.

## 5. 다른 예제를 쓸 때

`--check` 출력의 `examples` 항목이 예제마다 빠진 것을 알려 줍니다.

| 예제 | 준비 |
| --- | --- |
| Shader Live | `.\.venv\Scripts\python.exe scripts/fetch_assets.py footage 10708812` |
| 무중력 과일 | `.\.venv\Scripts\python.exe -B scripts/setup_environment.py --install --with body`, `scripts/fetch_assets.py footage 7525471`, `scripts/fetch_assets.py pose-model` |
| 주방 글리치 | `scripts/fetch_assets.py footage 7525471`. Git을 설치할 때 함께 들어온 Git Bash에서 `bash run_glitch.sh`를 실행합니다. |

이 예제들은 `ffmpeg`와 `ffprobe`를 PATH에서 찾습니다. WinGet으로 설치했다면 새 터미널에서 바로 잡힙니다. MediaPipe와 MuJoCo는 Python 3.12에서 확인했습니다.
