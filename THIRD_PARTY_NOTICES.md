# 출처와 별도 라이선스

이 저장소의 코드와 문서는 [MIT](LICENSE)입니다. 아래 외부 프로그램·라이브러리·영상·HDRI의 권리는 각 배포처에 있습니다.

## 이 저장소의 기원

- [hyun2xyz/threejs-colmap-drift](https://github.com/hyun2xyz/threejs-colmap-drift) commit `045c4c4a08ec65a5e03ee310325dd22ab45db8f6`의 내용을 같은 경로에 옮겨 담았습니다. 해당되는 것은 Drift 예제, `Water/`, `tracking/`, `examples/backboard_water`, `docs/pipeline`입니다. 2026-11-11에 같은 저장소의 commit `8ac31c6`에서 Shader Live·주방 글리치·무중력 과일 예제, 브레이크다운 제작 도구, OBS 메모, 예제 영상 받기 도구를 더 옮겼습니다. p5.js(LGPL-2.1)를 쓰는 활주로 두들 예제만 넣지 않았습니다. 주방 글리치 예제는 그 뒤 패키지에서 뺐습니다.
- `pipeline/`과 `shots/`는 2026-11-06 MacBook 테스트(외장 SSD 작업 폴더)의 Codex 세션 스크립트를 옮겨 정리한 것입니다. 실내 물, 정원 공, 채석장 천, 채석장 재조명이 여기에 해당합니다. 경로를 숏 설정 기반으로 바꾸고, COLMAP CLI 단계를 pycolmap으로 바꾸고, Windows에서 동작하도록 심볼릭 링크를 복사로 바꿨습니다.
- `scripts/jobs.py`와 학생용 문서 구성은 [hyun2xyz/AFTEREFFECT_COLMAP](https://github.com/hyun2xyz/AFTEREFFECT_COLMAP)(MIT, 같은 저작자)에서 가져왔습니다.

## 영상과 HDRI (저장소에 포함하지 않음)

`python -m pipeline fetch`가 원 배포처에서 직접 받고 SHA-256을 대조합니다.

| 숏 | 원본 | 제작자 | 라이선스 |
| --- | --- | --- | --- |
| `hall_water` | [Drone Video of the Interior of an Abandoned Building](https://www.pexels.com/video/drone-video-of-the-interior-of-an-abandoned-building-13939575/) | Florian Delée | [Pexels License](https://www.pexels.com/license/) |
| `courtyard_spheres` | [Drone Footage of a Courtyard](https://www.pexels.com/video/drone-footage-of-a-courtyard-9125189/) | Be The Observer | Pexels License |
| `quarry_cloth` | [Aerial View of Green and Rocky Quarry Landscape](https://www.pexels.com/video/aerial-view-of-green-and-rocky-quarry-landscape-33990833/) | David Pickup | Pexels License |
| `garage_spheres` | [A Push-in Shot in a Multi Level Carpark](https://www.pexels.com/video/a-push-in-shot-in-a-multi-level-carpark-4059360/) | Steve B | Pexels License |

`examples/`의 영상은 `scripts/fetch_assets.py`가 Pexels 파일 서버에서 직접 받습니다.

| 예제 | 원본 | 제작자 | 라이선스 |
| --- | --- | --- | --- |
| `fruit_physics` | [Woman Dancing in the Kitchen](https://www.pexels.com/video/woman-dancing-in-the-kitchen-7525471/) | SHVETS production | Pexels License |
| `shader_live` 06 | [Top View of an Empty Basketball Court](https://www.pexels.com/video/top-view-of-an-empty-basketball-court-10708812/) | Devin Huynh | Pexels License |
| `shader_live` 08 | [Aerial View of Spacious Empty Parking Lot](https://www.pexels.com/video/aerial-view-of-spacious-empty-parking-lot-34696869/) | JIUN-JE LIN | Pexels License |
| `shader_live` 01 | [Road Through Sandy Desert at Sunrise](https://www.pexels.com/video/road-through-sandy-desert-at-sunrise-10872797/) | Atif Dar | Pexels License |

Pexels License는 무료 다운로드·사용·수정을 허용하고, 출처 표기는 선택입니다. 다만 CC0는 아닙니다. 수정하지 않은 원본을 판매하거나 스톡 플랫폼에 재배포하는 것, 제작자나 배포처의 보증·후원을 암시하는 것은 금지됩니다. 그래서 원본 영상은 저장소에 넣지 않습니다. `docs/pipeline/*.webp`는 합성 결과의 스틸입니다.

HDRI는 Poly Haven의 [CC0](https://polyhaven.com/license) 자료입니다(Powered by Poly Haven).

- [Abandoned Factory Canteen 01](https://polyhaven.com/a/abandoned_factory_canteen_01) — Sergej Majboroda
- [Autumn Forest 02](https://polyhaven.com/a/autumn_forest_02) — Andreas Mischok
- [Parking Garage](https://polyhaven.com/a/parking_garage) — Greg Zaal, Rico Cilliers

## 라이브러리와 도구

| 이름 | 쓰는 곳 | 라이선스 |
| --- | --- | --- |
| [pycolmap / COLMAP](https://github.com/colmap/colmap) | SIFT, 매칭, 증분 복원, 번들 조정 | BSD-3-Clause |
| [ModernGL](https://github.com/moderngl/moderngl) | 네이티브 OpenGL 렌더 | MIT |
| [OpenCV](https://opencv.org/) | LK 광류, PnP, 이미지 입출력 | Apache-2.0 |
| [NumPy](https://numpy.org/), [SciPy](https://scipy.org/) | 수치 계산 | BSD-3-Clause |
| [three.js](https://threejs.org/) | Node에서 MarchingCubes 표면 | MIT ([Water/THREE-LICENSE.txt](Water/THREE-LICENSE.txt)) |
| [Pillow](https://python-pillow.org/) | 브레이크다운 자막 | MIT-CMU |
| [MediaPipe](https://github.com/google-ai-edge/mediapipe) | 몸 추적 (`fruit_physics`, 선택 설치) | Apache-2.0 |
| [MuJoCo](https://mujoco.org/) | 강체 물리 (`fruit_physics`, 선택 설치) | Apache-2.0 |
| [FFmpeg](https://ffmpeg.org/) | 디코드·인코드 (별도 설치, 저장소에 포함하지 않음) | LGPL/GPL (빌드에 따라 다름) |

## 모델 (저장소에 포함하지 않음)

| 항목 | 출처 | 조건 |
| --- | --- | --- |
| MediaPipe Pose Landmarker 모델 | https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker | 모델 카드의 조건을 따릅니다. `scripts/fetch_assets.py pose-model`로 받습니다. |
| SAM 3D Body (선택) | https://github.com/facebookresearch/sam-3d-body | SAM License. 가중치는 Hugging Face에서 접근 신청과 승인이 필요합니다. 이 저장소에는 코드도 가중치도 없습니다. [안내](docs/sam-3d-body.md) |

## 참고한 기법

`examples/shader_live`의 GLSL은 Shadertoy에서 널리 쓰이는 기법 계열(파동 옥타브 바다, 도메인 워핑, 보로노이 경계 거리 등)을 참고해 새로 썼습니다. 다른 사람의 셰이더 코드를 복사하지 않았습니다. 작품별 참고는 각 `NOTES_KO.md`에 있습니다.

## 쓰거나 고쳐서 배포할 때

`LICENSE`의 저작권 표시(`Copyright (c) 2026 hyun2xyz`)와 허가문을 함께 두면 됩니다. 코드와 모델 가중치, 영상의 라이선스는 서로 다릅니다. 공개하거나 상업적으로 쓰기 전에 실제로 쓴 버전과 자산의 조건을 확인하세요.
