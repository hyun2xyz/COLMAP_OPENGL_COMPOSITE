# Space Distortion · 팔이 지나간 공기가 유리처럼 휘는 효과

춤추는 사람의 팔이 지나간 자리의 **공기**가 뜨거운 유리처럼 휘어 뒤 배경과 소매가 끌려가는 합성입니다.
COLMAP으로 카메라를 풀고, MediaPipe로 팔 궤적을 잡아 그 궤적을 **장면 안 3D 위치에 남기고**, 네이티브 OpenGL(ModernGL)로
굴절·끌림·**프리즘 무지개**·유리 반사를 렌더합니다. 카메라가 움직여도 휘어진 공기는 팔이 지나간 그 자리에 붙어 있고,
강한 동작에서는 몸에서 **음파 같은 물결**이 부위별 곡선을 따라 여러 방향으로 퍼져 화면 끝과 카메라 앞 바닥까지 오고,
장면 속 **3D 연기(GPU 유체 시뮬레이션)** 가 팔에 휘저어지고 음파에 실려 카메라 앞까지 날아오거나 뒤로 밀려갑니다.

## 소스

- Pexels 32109503 "Contemporary Dance Solo Outdoors", 촬영 Anh Nguyen, [Pexels License](https://www.pexels.com/license/).
  페이지: https://www.pexels.com/video/contemporary-dance-solo-outdoors-32109503/
- 1920×1080, 25fps, 163프레임(6.52초). 유리 외벽과 콘크리트 바닥 앞, 카메라는 천천히 팬하며 약 10cm 이동합니다.
- 두 번째 예제 `piece_tunnel/`: Pexels 13607734 "Dancer in Tunnel", 촬영 khanhhoangminh, Pexels License.
  페이지: https://www.pexels.com/video/dancer-in-tunnel-13607734/ . 1920×1080, 24fps, 원본 377프레임 중 80번부터 216프레임(9초).
  그래피티 터널, **삼각대 고정 카메라**입니다.
- 세 번째 예제 `piece_street/`: Pexels 5359146 "Young Woman Dancing and Lip Synching", 촬영 Sergey Makashin, Pexels License.
  페이지: https://www.pexels.com/video/young-woman-dancing-and-lip-synching-5359146/ . 1920×1080, 23.976fps, 앞 200프레임(8.3초).
  가로 줄무늬 외벽 앞 거리, 카메라 고정. 긴 옷차림의 힙합 댄서입니다.
- 영상은 저장소에 넣지 않습니다(`footage/`는 git 무시). `python scripts/fetch_assets.py footage 32109503 13607734 5359146`로 받습니다.

## 준비

저장소 루트에서 실행합니다. `PY`는 Windows에서 `.venv\Scripts\python.exe`, Mac에서 `.venv/bin/python`입니다.

```sh
PY scripts/setup_environment.py --install --with body     # MediaPipe가 없으면
PY scripts/fetch_assets.py footage 32109503
PY scripts/fetch_assets.py pose-model                     # fruit_physics와 같은 MediaPipe heavy 모델을 함께 씁니다
PY scripts/setup_environment.py --check                   # examples.space_distortion.ready = true
```

## 실행

```sh
PY examples/shader_live/pipeline/slv.py prep  examples/space_distortion/piece   # 프레임 추출
PY examples/space_distortion/track_pose.py                                       # 포즈 + 사람 마스크 + COLMAP용 가림 마스크
PY examples/shader_live/pipeline/slv.py solve examples/space_distortion/piece   # COLMAP (사람·그림자는 mask_dir로 제외), 약 1.5분
PY examples/shader_live/pipeline/slv.py plane examples/space_distortion/piece   # 바닥 평면 → work/check_plane.jpg 를 꼭 열어 봅니다
PY examples/space_distortion/trails.py                                           # 팔을 장면 좌표로 → work/check_limbs.jpg
PY examples/space_distortion/render.py --still 30,90,150                         # 스틸 세 장 → work/still_sheet.jpg
PY examples/space_distortion/render.py                                           # 전체 렌더 + 인코딩
PY examples/shader_live/pipeline/slv.py track examples/space_distortion/piece   # 브레이크다운용 트래킹 패스
PY examples/shader_live/pipeline/breakdown.py examples/space_distortion/piece   # 1080×1350 브레이크다운
PY examples/space_distortion/breakdown_passes.py                                 # 패스 브레이크다운(두 예제를 모두 렌더한 뒤)
PY examples/space_distortion/breakdown_rules.py examples/space_distortion/piece  # 제작 규칙(docs/breakdown v7)대로 만든 최종 브레이크다운
```

터널·거리 예제는 위 명령에서 `piece`를 `piece_tunnel`·`piece_street`로 바꿔 똑같이 돌립니다(`track_pose.py`, `trails.py`, `render.py`도 첫 인자로 폴더를 받습니다).
단, 카메라가 움직이지 않아 COLMAP은 시차가 없어 초기화에 실패합니다. 그래서 `piece.json`에 `"solver": "homography"`, `"tilt": 90`
(카메라가 수평: 터널 소실점이 화면 가운데 높이)을 두어 Shader Live의 호모그래피 풀이를 씁니다. 216프레임 내내 카메라 이동은 0으로 나옵니다.

결과(`piece/out/`): `Space_Distortion_Composite.mp4`, `Space_Distortion_Plate.mp4`,
`Space_Distortion_Compare_1080x1350.mp4`(위 효과 / 아래 원본, 레퍼런스와 같은 배치), 브레이크다운,
`Space_Distortion_Passes_Breakdown_1080x1350.mp4`(두 예제를 위아래로 쌓고 세로 와이프로 패스를 넘김),
각 예제의 `*_Passes_Grid_1920x540.mp4`(일곱 패스를 한 화면에).
제작 규칙을 따른 최종 브레이크다운은 `*_Breakdown_Final_1080x1350.mp4`입니다([docs/breakdown](../../docs/breakdown/README.md): 흰 바탕, 21px 네모 캡션, RESULT/ORIGINAL 3초 → 01 추적점+바닥 격자 3초 → 3D 보기 → 02~07 한 재생 시계 위 오른쪽→왼쪽 9프레임 와이프 → RESULT 0번 1초 정지 → 결과 전체 재생). 편집 결정과 캡션 위치는 같은 이름의 `.json`에 남깁니다.

## 확인한 것 (2026-11-18, Windows · RTX 4070 Ti SUPER)

| 단계 | 결과 |
| --- | --- |
| MediaPipe | 163/163 프레임 검출 |
| COLMAP | 163/163 등록, 평균 재투영 0.52px(반 해상도), 점 656개, 초점 1844px(가로 화각 55° 가정) |
| 바닥 평면 | `level` 가정(롤 없음), 지평선 아래 1.4°. `check_plane.jpg`에서 격자가 세 프레임 모두 바닥에 붙어 있음 |
| 축척 | 카메라 높이 0.96m 가정 → 코 높이 중앙값 1.61m(약 1.7m 사람), 사람까지 3.6~3.9m |
| 터널(`piece_tunnel`) | MediaPipe 216/216, 호모그래피 풀이 인라이어 중앙값 1772, 카메라 이동 0. 가로 화각 65°·카메라 높이 0.9m 가정 → 코 높이 1.61m, 사람까지 3.8~4.0m |
| 거리(`piece_street`) | MediaPipe 200/200, 호모그래피 풀이 인라이어 최소 228·중앙값 258(바닥이 좁은 띠), 카메라 이동 0. 가로 화각 60°·카메라 높이 1.0m 가정 → 코 높이 1.57m, 사람까지 3.4~4.1m. 음파 5번(55·104·138·158·183프레임) |
| 물결·프리즘 (2026-11-19) | 물결 고리 6개씩(기존 2·23·44·115·132·150, 터널 15·35·55·150·169·188프레임). 아베수 3 → 400nm는 왜곡 ×1.49, 700nm는 ×0.87. 확인은 스틸 표본만 함: 무지개 띠가 물결 마루와 카메라 앞 바닥에 보이지만, 1m 넘게 퍼진 뒤의 고리는 단색 벽·바닥 위에서 약하게 보임 |

## 어떻게 만드는가

1. **마스크를 둔 COLMAP.** 춤추는 사람과 바닥 그림자는 움직이므로 특징점에서 뺍니다. `track_pose.py`가 MediaPipe 마스크를 부풀리고
   발 오른쪽(그림자 쪽) 상자를 더해 `work/solve_mask/`에 쓰고, `piece.json`의 `mask_dir`로 Shader Live `slv.py solve`에 넘깁니다.
2. **팔을 장면에 올리기.** 가장 낮은 발 화소의 광선이 바닥 평면과 만나는 곳이 사람까지의 거리입니다. 관절마다 자기 2D 화소를 그대로 쓰고
   그 거리 + MediaPipe의 상대 깊이를 줍니다. 그래서 화면에서는 관절과 정확히 겹치고, 3D에서는 바닥 위 제자리에 섭니다.
3. **공기 리본.** `trail.limbs`의 팔 사슬(팔꿈치-손목-검지, 어깨-팔꿈치-손목)이 지난 `age_frames`프레임 동안 지나간 면을 리본 메시로 만듭니다.
   위치는 장면 좌표라 **이번 프레임의 카메라**로 투영됩니다. 조각마다 (움직임 방향 × 세기, 세기)를 반 해상도 float 버퍼에 더합니다.
   세기는 속도(느린 팔은 흔적이 없음) × 나이 감쇠 × 리본 가장자리 감쇠입니다. 손 끝 쪽은 `trail.tip`만큼 더 뻗습니다.
4. **합성.** 버퍼를 흐리게 한 뒤, 플레이트를 `움직임 방향으로 끌기 + 세기 기울기로 렌즈처럼 휘기 + fbm 아지랑이`만큼 옮겨 읽습니다.
   7탭 끌림, 파장별 갈라짐(8번 프리즘), 기울기가 급한 곳의 어두운 테두리(`edge_dark`)와 한쪽 하이라이트가 유리 느낌을 냅니다.
   지금 프레임의 사람 마스크에서는 `body_keep`만큼 약하게 해서 팔과 몸이 읽히게 둡니다.
   **몸 주변에만**: 궤적 장은 현재 실루엣에서의 거리로 줄어듭니다(`near_m`, 가우스). 팔이 떠난 자리의 공기는 몸에서 멀어질수록 옅어져서,
   허공에 따로 떠 있는 물결 막처럼 보이지 않고 몸을 감싼 뜨거운 공기처럼 보입니다. 아지랑이 노이즈도 더 잘게(움직임 방향으로 흘러가게) 했습니다.

5. **윤곽선 파동("팡 팡!").** 손목·팔꿈치·무릎이 빠르게 움직이다 3프레임 안에 멈추는 순간(속도 감소가 `shock.joints`의 px/프레임 이상)을
   타격으로 봅니다. 타격마다 `shock.pulses`번(`gap_frames` 간격) 그 관절 근처의 **몸 윤곽선을 따라** 공기 띠가 바깥으로 밀려 나갑니다.
   현재 사람 마스크에서 바깥쪽 거리장을 만들고, 거리 = `radius_m`까지 퍼지는 앞면에 띠를 둡니다. 띠는 관절에서 `reach_m`(+퍼진 거리)까지만
   윤곽을 따라 번지고, 관절 위치는 장면 좌표라 카메라가 움직여도 제자리입니다. 이 띠는 팔 궤적과 **같은 왜곡 장**에 더해지므로
   끌림·렌즈 휨·아지랑이·프리즘·반사가 손 궤적과 똑같은 모양으로 나옵니다.
6. **유리 반사.** 휘어진 공기(궤적 장 + 윤곽선 파동)를 높이 면으로 보고 법선을 구해, 투명한 유리처럼 비춥니다. 정면은 그대로 비치고
   (`lift`만큼 조금 밝게), 비스듬한 테두리는 법선 방향 플레이트를 반사(`reflect`, 빛을 **더하기만** 해서 어두운 막이 생기지 않음)하며,
   왼쪽 위 해 방향의 하이라이트(`spec`)가 맺힙니다.

7. **몸에서 나가는 음파.** 가장 강한 타격(`ring.min_hit` 이상, `ring.gap_frames`마다 하나)마다 물결이 `ring.echoes`번
   (`ring.echo_frames` 간격) 나갑니다. 화소마다 3D 점이 있어야 거리를 재므로 장면을 대역 면으로 둡니다: **바닥(z=0)** 과
   **사람을 지나는 세운 벽**(카메라를 마주 봄, 엉덩이에서 `wall_back_m` 뒤). 두 면은 부드러운 모서리(광선 거리의 smooth min)로 이어 붙여 이음선이 없습니다.
   - **앞면 모양 = 몸.** 물결 앞면은 한 점에서의 거리가 아니라, 타격 순간의 **몸(MediaPipe 관절 사이 캡슐 15개: 팔·손·다리·어깨·골반·몸통·머리)까지의 거리**입니다.
     캡슐 거리는 지수 smooth-min으로 합쳐 꺾임이 없고, 때린 관절에서 먼 부위는 `lag_m`만큼 늦게 출발합니다(`focus_m` 가우스).
     그래서 처음에는 때린 팔에서 나와 팔·몸통·다리 윤곽을 곡선 곡선 따라 감싸고, 퍼질수록 둥글어집니다.
   - **유기적으로.** 퍼지는 거리가 방향마다 3D 노이즈로 ±`wobble` 흔들려 멀리 가도 원이 되지 않습니다.
   - **여러 방향 무작위.** 물결마다 무작위 방향 둘(von Mises 꽃잎)로 세기가 달라 한쪽으로 쏠려 나가고, 속도도 ±20% 다릅니다(`ring.seed`로 고정).
   - 바닥에서는 화면 아래(카메라 쪽)로 쓸려 오며 원근 때문에 가까울수록 넓고 세지고, 벽에서는 사람 둘레로 열려 화면 밖까지 나갑니다.
     단면은 `sin(2.6x)·exp(-0.45x²)`(마루 몇 개), 밀리는 방향은 앞면의 법선(화면에서의 거리 기울기)입니다.
8. **프리즘(무지개).** 유리는 짧은 파장을 더 꺾습니다. 코시 식 n(λ) = A + B/λ²의 A, B를 n_d = 1.5와 **아베수**(`prism.abbe`)로 정하고,
   400–700nm 파장 `prism.samples`개가 각각 왜곡량 × (n−1)/(n_d−1)만큼 옮겨 읽습니다. 읽은 값은 그 파장의 색(CIE 1931 등색 함수를
   Wyman·Sloan·Shirley 2013의 해석식으로 계산 → 선형 sRGB, 채널마다 합이 흰색이 되게 정규화)으로 더합니다. 그래서 휘는 곳의 밝은 경계가
   파장별로 갈라져 무지개 테가 생깁니다. 실제 유리는 아베수 약 60이라 거의 안 보이므로 3으로 과장합니다.
   또 휘어진 공기를 기울어진 유리로 보고, 파장마다 법선 기울기를 (n−1)만큼 달리해 해 방향 하이라이트(`prism.glint`)를 구합니다.
   하이라이트가 파장마다 조금씩 다른 자리에 맺혀 물결 마루에 작은 무지개 띠가 됩니다.
9. **브레이크다운 패스.** 합성 셰이더가 같은 계산에서 패스를 함께 냅니다(`work/pass/`). **Warp Grid**는 원본 위 격자를 같은 왜곡으로 읽은 것,
   **Distortion Field**는 옮겨지는 거리(px)를 열화상 색으로 칠한 것(matplotlib `inferno`, 없으면 Google Turbo),
   **Normal Map**은 유리 면 법선(평평한 곳이 보라색 (0.5, 0.5, 1)), **Prism Split**은 프리즘이 더한 색(분광 결과 − 단색 결과)입니다.
   **3D Smoke**는 연기를 깊이 색으로 칠한 것입니다(주황 = 카메라 가까이, 파랑 = 멀리).
   `breakdown_passes.py`가 레퍼런스처럼 세로 선이 지나가며 패스를 넘기는 영상으로 엮습니다.
10. **3D 연기(`smoke.py`).** Stam의 Stable Fluids를 GPU에서 푸는 방식(Harris, GPU Gems 38장)을 Pavel Dobryakov의 WebGL-Fluid-Simulation(MIT)
   셰이더 구조를 따라 **3D로** 옮겼습니다: 힘 → 와도(curl) → 와도 유지(vorticity confinement) → 발산 → Jacobi 압력 → 기울기 빼기 → 이류.
   - **장면 속 상자.** 격자는 평면 좌표(미터) 안의 상자입니다. 첫 프레임 카메라에 맞춰 놓고, 렌즈 앞 `near_m`(0.3m)부터 사람 뒤 `back_m`(3m)까지,
     폭 `width_m`, 높이 `height_m`입니다. 기본 112×56×128칸(약 5cm)이고, 영상 한 프레임에 두 번 풉니다.
   - **OpenGL 3.3 그대로.** macOS는 컴퓨트 셰이더가 없으므로 볼륨을 2D 아틀라스(깊이 칸마다 한 타일, `tiles_x`개씩)에 두고, 모든 단계를 아틀라스 전체에
     그리는 조각 셰이더로 풉니다. 삼선형 보간은 타일 안 쌍선형 + 두 타일 사이 보간입니다.
   - **팔이 휘젓기:** 팔꿈치·손목·손끝·무릎·발목(MediaPipe를 3D로 올린 것) 주변 공기가 관절 속도로 끌려가고, 빠를수록 연기가 조금 나옵니다(`arm_emit`).
   - **음파가 앞뒤로 불어내기:** 7번과 같은 물결 앞면(타격 순간 몸까지의 거리)을 3D로 계산해, 앞면과 그 뒤쪽 공기를 앞면 법선 방향 바깥으로 밉니다(`wave_push_mps`).
     타격마다 첫 물결은 꽃잎 하나를 **카메라 쪽**, 하나를 **반대쪽(장면 뒤)** 으로(±35° 무작위) 둡니다. 타격 순간 때린 관절 둘레에 연기가 한 번 뿜어지고(`burst`),
     물결이 그것을 렌즈 앞까지, 또 장면 뒤로 실어 갑니다.
   - **몸은 장애물:** 지금 프레임의 관절 사이 캡슐 15개 안에서는 속도가 줄고 연기가 밀려납니다.
   - **바닥 연기:** 바닥 근처(`ground_m`)에 노이즈 모양 조각으로 조금씩 나오고(`ambient`), 부력(`buoyancy`)으로 오릅니다.
   - **렌더:** 반 해상도에서 화소마다 이 프레임의 카메라 광선을 상자 안으로 쏘아 앞에서부터 쌓습니다(`step_m`, 멀수록 넓게).
     칸 크기보다 작은 뭉게뭉게한 모양은 장면 좌표의 3D 노이즈(`detail_per_m`)로 밀도를 깎아 냅니다. 빛은 해 방향으로 밀도를 짚은
     투과율 볼륨(자기 그림자)으로 넣습니다. 광선은 바닥에서 멈추고, 사람 마스크 안에서는 사람 깊이에서 멈춥니다.
     그래서 **사람 뒤 연기는 가려지고 앞 연기는 사람을 덮으며**, 렌즈 가까이 온 연기는 크고 부드럽게, 멀리 간 연기는 작게 보입니다.
   - 합성에서는 연기를 **왜곡을 거쳐** 읽습니다(같은 공기 속에 있으므로 물결·궤적에 함께 휨).

## 가져다 쓴 공개 알고리즘

| 무엇 | 출처 | 사용 |
| --- | --- | --- |
| CIE 1931 등색 함수 해석식 | C. Wyman, P.-P. Sloan, P. Shirley, "Simple Analytic Approximations to the CIE XYZ Color Matching Functions", JCGT 2(2), 2013, https://jcgt.org/published/0002/02/01/ | 식(다중 가우스 맞춤)과 계수를 `render.py spectral()`에 옮겨 씀. 1nm 합이 표값 ≈106.9와 맞는지 확인 |
| 코시 분산식, 아베수 | 광학 교과서 식 | 파장별 굴절률 |
| XYZ → 선형 sRGB 행렬 | IEC 61966-2-1 | 파장 색 |
| Turbo 색표 다항 근사 | Copyright 2019 Google LLC, Apache-2.0 (A. Mikhailov, Ruofei Du), https://gist.github.com/mikhailov-work/0d177465a8151eb6ede1768d51d476c7 | matplotlib가 없을 때 열화상 색 |
| inferno 색표 | matplotlib (N. Smith, S. van der Walt, CC0) | 열화상 색(설치되어 있으면) |
| Stable Fluids | J. Stam, "Stable Fluids", SIGGRAPH 1999 / M. Harris, "Fast Fluid Dynamics Simulation on the GPU", GPU Gems 38장, 2004 | 연기 풀이 방식 |
| 유체 셰이더 구조 | Pavel Dobryakov, WebGL-Fluid-Simulation, MIT License (c) 2017, https://github.com/PavelDoGreat/WebGL-Fluid-Simulation | curl·vorticity·divergence·pressure·gradient·advection 셰이더를 GLSL 3.30으로 옮김(`smoke.py` 머리말에 저작권 표기) |

Shadertoy의 분산 셰이더들은 기본 라이선스가 CC BY-NC-SA라 코드를 가져오지 않고, 위 논문 식으로 직접 썼습니다.

## piece.json 의 효과 값

| 키 | 뜻 |
| --- | --- |
| `trail.limbs`, `limb_gain` | MediaPipe 관절 번호 사슬과 각 사슬의 세기 |
| `trail.age_frames`, `sub`, `fade_in`, `tip` | 흔적 길이(프레임), 프레임당 시간 샘플, 손 바로 뒤를 비우는 비율, 손끝 연장 비율 |
| `distort.strength`, `ripple`, `haze`, `gain` | 끌림(px), 렌즈 휨, 아지랑이, 세기 → 덮임 변환 |
| `distort.edge_dark`, `body_keep` | 테두리 어둡기, 몸 위에서 줄이는 비율 |
| `distort.shock`, `reflect`, `spec`, `lift` | 윤곽선 파동 세기(궤적 장에 더하는 비율), 테두리 반사, 하이라이트, 유리 안쪽 밝기 |
| `distort.near_m` | 궤적 왜곡이 남는 몸 윤곽에서의 거리(m). 작을수록 몸에 붙음 |
| `prism.abbe`, `samples` | 아베수(작을수록 무지개가 넓음, 0이면 끔), 파장 샘플 수(최대 12) |
| `prism.glint`, `tilt` | 파장별 하이라이트 세기, 왜곡(px)을 유리 기울기로 바꾸는 비율 |
| `ring.amp_px`, `decay`, `speed_mps`, `life_s` | 물결 고리의 밀림(px, 사람 거리 기준·가까울수록 최대 3배), 사라지는 곡선, 퍼지는 속도, 수명 |
| `ring.width_m`, `grow`, `min_hit`, `gap_frames`, `wall_back_m` | 고리 폭과 퍼질수록 넓어지는 비율, 고리를 낼 타격 세기, 고리 사이 최소 간격, 대역 벽 위치 |
| `ring.echoes`, `echo_frames`, `lag_m`, `focus_m`, `wobble`, `seed` | 타격당 물결 수와 간격, 먼 부위가 늦게 출발하는 거리, 때린 관절 주변 범위, 방향별 흔들림, 무작위 씨앗 |
| `smoke.enable`, `grid`, `tiles_x`, `substeps`, `pressure_iter` | 연기 켜기, 칸 수(가로·높이·깊이), 아틀라스 한 줄의 타일 수(깊이 칸 수의 약수), 프레임당 풀이 횟수, 압력 반복 |
| `smoke.width_m`, `height_m`, `near_m`, `back_m` | 상자 폭·높이, 렌즈 앞 시작 거리, 사람 뒤로 더 가는 거리 |
| `smoke.vorticity`, `vel_diss`, `dye_diss`, `buoyancy` | 소용돌이 유지(1 아래. 크면 격자 노이즈가 폭주), 속도·연기 사라짐, 부력 |
| `smoke.arm_push`, `arm_radius_m`, `arm_emit`, `wave_push_mps`, `burst` | 관절이 공기를 끄는 세기와 범위, 관절에서 나오는 연기, 물결이 부는 바람(m/s, 지나는 프레임마다), 타격 때 뿜는 연기 |
| `smoke.ambient`, `ground_m`, `preroll` | 바닥 연기 양과 높이, 시작 전 미리 돌리는 프레임 |
| `smoke.sigma`, `step_m`, `detail_per_m`, `color`, `ambient_light`, `sun_light` | 밀도당 소광(1/m), 광선 간격, 뭉게 노이즈 촘촘함(1/m), 색(sRGB), 주변광, 햇빛(해 방향은 `sun`) |
| `breakdown.compare_src`, `process_start_src`, `slow` | `breakdown_rules.py` 편집: 처음 비교에 틀 원본 프레임 [시작, 끝], 과정 단계의 시작 원본 프레임, 느리게 길게 보일 단계 {이름: [배속, 출력 프레임 수]}(캡션에 `/ 0.5x` 표기). 없으면 원본 0부터 3초, 모든 단계 1배속 1초 |
| `shock.joints`, `pulses`, `gap_frames` | 타격으로 볼 관절과 속도 감소 기준(px/프레임), 타격당 고리 수와 간격 |
| `shock.radius_m`, `tau_s`, `life_s`, `reach_m` | 윤곽에서 퍼지는 거리(m), 퍼지는 빠르기, 사라지는 시간, 윤곽을 따라 번지는 범위(m) |

## 한계

- 초점거리(가로 55°), 바닥 높이 축척(카메라 0.96m), 바닥 수평은 가정입니다. 카메라가 거의 팬만 하므로 시차가 작습니다.
- 사람 깊이는 바닥에 닿은 발 하나와 MediaPipe 상대 깊이에서 나옵니다. 점프 구간에는 맞지 않습니다.
- 손가락 관절은 MediaPipe Pose의 검지·새끼 점뿐입니다. 소매 끝 모양은 따라가지 않습니다.
- 효과는 사람 앞뒤를 구분하지 않습니다(휘어진 공기가 몸 위에도 그려짐). `body_keep`으로 약하게 할 뿐입니다.
- 물결 고리의 3D 위치는 바닥 + 세운 벽 대역 면 위에서 잽니다. 실제 뒤 벽(유리벽, 터널 옆벽)의 깊이는 쓰지 않으므로,
  벽에 맺히는 고리 크기는 사람 깊이에 있다고 본 값입니다.
- 무지개 세기는 물리값이 아니라 아베수 3으로 과장한 것입니다.
- 연기 칸은 약 5cm라 렌즈 바로 앞(1m 안)에서는 크게 흐려 보입니다. 세부 모양은 시뮬레이션이 아닌 노이즈입니다(흐름을 따라가지 않고 천천히 끓기만 함).
- 연기는 바닥에서만 멈추고 실제 벽(유리벽, 터널 벽)은 모릅니다. 사람 가림은 마스크 + 엉덩이 깊이 하나로 해서 팔이 몸 앞으로 많이 나오면 깊이가 조금 틀립니다.
