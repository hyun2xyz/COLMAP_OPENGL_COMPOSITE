# 처음 시작하기

이 폴더는 **Codex 또는 Claude Code와 함께** 쓰는 수업용 패키지입니다. 실사 영상의 카메라를 복원하고, 코드로 만든 물·공·천을 네이티브 OpenGL로 렌더해 합성합니다. 새 영상과 HDRI도 에이전트가 라이선스를 확인하며 직접 찾아 줍니다. Python·FFmpeg·Node가 없는 컴퓨터도 설치부터 시작할 수 있습니다. COLMAP 앱을 따로 설치할 필요는 없습니다(pycolmap이 함께 설치됩니다). 영상 원본은 저장소에 들어 있지 않고, 각 컴퓨터가 원 배포처에서 직접 받습니다.

## 1. 포크해서 받기

공통강의의 Git 세팅(Git·GitHub CLI 설치와 `gh auth login`)을 마친 컴퓨터에서 시작합니다. [포크로 시작하고 업데이트 받기](docs/fork-and-update.md) 1번에 있는 문장으로 이 저장소를 내 GitHub 계정에 포크해서 받은 뒤, 받은 폴더를 Codex나 Claude Code에서 엽니다. 전역 스킬은 설치하지 않아도 됩니다.

## 2. 폴더를 연 뒤 이 문장을 붙여 넣기

> 이 폴더의 AGENTS.md와 .agents/skills/opengl-composite/SKILL.md를 읽고 시작해 주세요.
> 실제 OS와 설치된 도구를 확인하고, 내 OS의 setup 문서에 있는 공식 출처로 빠진 것만 준비해 주세요.
> Python 3.13(또는 3.12)과 이 폴더의 .venv를 사용해 주세요.
> 먼저 예제 숏 hall_water를 소스 받기부터 OpenGL 합성까지 한 단계씩 진행하고,
> 단계마다 무엇을 확인했는지 알려 주세요. 원본과 기존 결과는 보존하고,
> 설치 완료·계산 완료·눈으로 확인한 것을 구분해서 말해 주세요.
> 긴 계산은 docs/background.md를 따르고, 실패를 성공으로 보고하지 마세요.

## 3. 진행 순서

1. **설치**: 내 OS의 문서 하나만 읽습니다. [Mac](docs/setup-macos.md) 또는 [Windows](docs/setup-windows.md)입니다. `scripts/setup_environment.py --check`가 `"ready": true`를 내면 준비가 끝난 것입니다. OpenGL 3.3 창 없는 컨텍스트까지 실제로 만들어 확인합니다.
2. **예제 숏 하나**: `python -m pipeline composite hall_water`를 실행합니다. 4초 숏 하나에 Apple M2 Max 기준 약 1분이 걸리고, 처음에는 영상 다운로드 시간이 더 듭니다. 끝나면 `work/hall_water/render/final/`의 검수 시트를 보고 `Composite.mp4`를 끝까지 재생합니다.
3. **내 영상**: "이런 장면에 이런 물체를 합성하고 싶어요"라고 말하면 에이전트가 [재료 찾기 규칙](docs/materials.md)에 따라 쓸 수 있는 영상과 HDRI를 찾아 후보를 보여 주고, `shots/`에 설정 파일을 만듭니다. 아무 영상이나 되는 것은 아닙니다. 하드컷 없이 카메라가 움직이고, 질감이 보이며, 줌이 고정된 짧은 숏이어야 합니다.

4. **다른 예제**: GLSL 효과를 장면에 고정하는 Shader Live, 사람 몸에 부딪히는 무중력 과일은 `examples/`에 있습니다. "Shader Live 예제를 해 보고 싶어요"처럼 말하면 됩니다. 필요한 것은 [README](README.md#파이프라인-밖의-예제)에 있습니다. 몸 추적은 MediaPipe로 바로 시작하고, SAM 3D Body를 쓰려면 [접근 신청](docs/sam-3d-body.md)을 미리 해 둡니다.

자동 검사가 통과해도 **영상을 끝까지 재생해서 본 것은 아닙니다.** 검수 시트와 실제 재생을 확인한 뒤에 완료로 기록합니다. 컴퓨터별로 실제로 확인한 범위는 [검증 범위](docs/validation.md)에 있습니다.

수업이 끝나면 내 작업을 내 포크에 push하고, 강사가 업데이트를 알리면 upstream에서 pull합니다. 방법은 [포크로 시작하고 업데이트 받기](docs/fork-and-update.md)의 2·3번에 있습니다.
