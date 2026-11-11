# Shader Live · 카메라를 풀고 GLSL 효과를 장면에 고정하기

실사 영상에서 카메라와 바닥 면을 풀고, 그 좌표계 안에서 GLSL 프래그먼트 셰이더 하나를 평가해 효과를 장면에 붙이는
예제입니다. 작품 세 개가 들어 있습니다.

| 작품 | 내용 | 카메라 풀이 | 영상 |
| --- | --- | --- | --- |
| `06_Marbled_Court` | 농구 코트 페인트 위로 번지는 마블링 잉크 | COLMAP | Pexels 10708812 |
| `08_Lava_Fissures` | 주차장 통로가 갈라지고 틈 아래 용암 | 평면 호모그래피 | Pexels 34696869 |
| `01_Desert_Causeway_Sea` | 사막 도로 옆 모래 평원이 바다로 | 평면 호모그래피 (지평선 기울기) | Pexels 10872797 |

각 작품 폴더에 `piece.json`(설정), `effect.glsl`(효과), `NOTES_KO.md`(만든 방법, 출처, 한계)가 있습니다.

## 준비

저장소 루트에서 실행합니다. 아래의 `PY`는 Windows에서 `.venv\Scripts\python.exe`, Mac에서 `.venv/bin/python`입니다.

```sh
PY scripts/setup_environment.py --check           # "ready": true, examples의 shader_live에 빠진 것이 없어야 합니다
PY scripts/fetch_assets.py footage 10708812 34696869 10872797
```

패키지는 저장소의 기본 설치(`setup_environment.py --install`)에 모두 들어 있습니다. `ffmpeg`와 `ffprobe`는 PATH에서 찾습니다.
OpenGL 3.3을 쓸 수 있는 그래픽 카드가 필요합니다. `python -m pipeline`의 숏과는 별개의 예제라서 결과는 `work/`가 아니라
각 작품 폴더의 `work/`와 `out/`에 생깁니다.

## 한 작품 만들기

```sh
PY examples/shader_live/pipeline/slv.py prep  examples/shader_live/pieces/06_Marbled_Court    # 프레임 추출
PY examples/shader_live/pipeline/slv.py solve examples/shader_live/pieces/06_Marbled_Court    # 카메라 풀기 (COLMAP은 몇 분 걸립니다)
PY examples/shader_live/pipeline/slv.py plane examples/shader_live/pieces/06_Marbled_Court    # 평면과 좌표계
```

여기서 **반드시** `pieces/06_Marbled_Court/work/check_plane.jpg`를 엽니다. 첫·중간·마지막 프레임에서 흰 격자와 색 축이
같은 자리에 붙어 있어야 합니다. 등록 프레임 수와 오차가 좋아도 풀이가 틀린 경우가 있었습니다.

```sh
PY examples/shader_live/pipeline/render.py examples/shader_live/pieces/06_Marbled_Court --still 60,150,209   # 스틸 세 장 → work/still_sheet.jpg
PY examples/shader_live/pipeline/finish.py examples/shader_live/pieces/06_Marbled_Court                      # 전체 렌더 + 브레이크다운
PY examples/shader_live/pipeline/verify_sheet.py local-runs/verify.jpg 06_Marbled_Court                      # 인코딩본에서 네 프레임 확인
```

결과는 `pieces/<작품>/out/`에 생깁니다: `*_Composite.mp4`, `*_Plate.mp4`, `*_Straight_Alpha.mov`,
`*_Breakdown_1080x1350.mp4`. `work/finish.log`의 마지막 줄이 `FINISHED_OK`인지 확인합니다.

좌표를 찍는 도구:

```sh
PY examples/shader_live/pipeline/slv.py pick   <작품 폴더> 0 960 540            # 그 프레임의 화소 아래 평면 좌표(m)
PY examples/shader_live/pipeline/slv.py pick3d <작품 폴더> 0 972 433 209 951 478  # 두 프레임에서 같은 점을 찍어 3D 위치 구하기
```

## piece.json

| 키 | 뜻 |
| --- | --- |
| `source`, `start`, `frames`, `step` | 원본 경로(작품 폴더 기준), 시작 프레임, 프레임 수, 몇 프레임마다 1장(60fps 원본은 2) |
| `hfov` | 가정한 가로 화각(기본 73.7°). 스톡 영상에는 렌즈 정보가 없습니다. |
| `solver` | 없으면 COLMAP. `"homography"`면 평면 호모그래피 솔버 |
| `tilt` | 호모그래피 솔버 전용. 첫 프레임이 수직 아래에서 몇 도 기울었는지(수직 하강 샷은 0) |
| `refine_focal` | COLMAP이 초점거리를 보정할지. 입체감이 큰 선회 샷에만 켭니다. |
| `mask_roi` | 특징점을 뽑을 영역 [x0,y0,x1,y1]. 하늘, 구름, 태양 플레어, 물을 뺍니다. |
| `plane.roi`, `plane.anchor`, `plane.xdir`, `plane.cam_h` | 평면을 맞출 영역, 원점이 될 화소, +x 방향 화소, 카메라 높이(m, 축척을 정함) |
| `sun` | `{"az","el"}`(평면 좌표 기준 태양 방향) 또는 `{"px":[x,y]}`(화면 속 태양 화소) |
| `params` | 셰이더의 `uP[0..15]` |

## effect.glsl

`void effect(in vec3 ro, in vec3 rd, in vec2 px, inout FX fx)` 하나를 정의합니다. `ro`, `rd`는 평면 좌표
(미터, z = 0이 맞춘 평면)의 카메라 위치와 화소 시선입니다.

- `fx.over`, `fx.a`: 플레이트를 덮는 불투명 레이어
- `fx.add`: 더하는 빛
- `fx.mul`: 플레이트에 곱하는 값(그림자, 그레이딩)
- `fx.warp`: 플레이트를 읽는 위치의 화소 오프셋(굴절, 재투영)

합성식은 `mix(plate(px + warp) * mul, over, a) + add`입니다. `plate(px)`, `project(P)`, `hitPlane(ro, rd, h)`,
`vnoise`, `fbm`, `uSun`, `iTime`, `uDur`를 쓸 수 있습니다.

## 내 영상으로

1. 6~8초 구간을 고릅니다. 카메라가 움직이고, 넓은 바닥이나 벽이 보이고, 장면이 정지해 있는 영상이 좋습니다.
2. 작품 폴더를 복사해 `piece.json`의 `source`, `frames`, `plane`을 바꿉니다.
3. 입체감이 있는 선회 샷은 COLMAP, 수직 하강 샷이나 평평한 도로는 `"solver": "homography"`로 풉니다.
4. `check_plane.jpg`가 맞을 때까지 솔버와 `mask_roi`를 고칩니다. 그다음에 셰이더를 씁니다.

## 한계

- 초점거리와 길이 단위는 가정입니다. 평면 위의 고정은 가정과 무관하게 맞지만, 평면 위로 솟은 그래픽은 가정만큼 어긋날 수 있습니다.
- 화소별 깊이가 없습니다. 나무나 사람이 효과 앞을 지나가는 영상에는 그대로 쓸 수 없습니다.
- `*_Straight_Alpha.mov`는 그 플레이트 전용입니다. 다른 배경용 소재가 아닙니다.
- Windows의 NVIDIA 그래픽 카드에서 확인했습니다. Mac에서는 확인하지 못했습니다.
