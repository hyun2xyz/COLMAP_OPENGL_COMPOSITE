# 3D Liquid Glass & Fluted Optics · 실사 3D 공간의 유체 유리·플루티드 글래스

COLMAP으로 푼 실사 3D 카메라 경로와 바닥 평면(`z = 0`) 위에 **3D 리퀴드 글래스 메타볼 군집 2개**와 **세로 골(Fluted/Reeded) 건축 유리 모노리스 1개**를 월드락 배치해, 네이티브 OpenGL(ModernGL, GLSL 3.30) 레이마칭으로 합성하는 테스트 예제입니다.

- **전방 리퀴드 글래스 군집 (`cluster1`):** 햇살이 드는 복도 초입 우측에 떠 있으며 카메라가 바로 옆을 스쳐 지나갑니다.
- **플루티드 글래스 모노리스 (`flute_pos`):** 중앙 기둥과 창문 사이에 떠 있으며, 기둥·창틀을 세로 광학 홈으로 왜곡하고 바닥에 평행 카우스틱 띠를 투사합니다.
- **중앙 홀 메인 리퀴드 글래스 (`cluster2`):** 거실 창문 앞 공중에 떠서 뭉치고 갈라지며 바닥에 굴절 카우스틱을 남깁니다.
- **고밀도 갤러리 확장본 (`render_glass_dense.py` → `piece_glass/out_dense_v2/`):** 13.4m 복도 전 구간에 리퀴드 글래스 메타볼 군집 5개 + 세로 골 플루티드 모노리스 4개 + 188개 위성 방울을 배치하고, **COLMAP 삼각측량으로 맞춘 6개 투명 3D 건축 홀드아웃(좌·우 입구 벽 + 구조 기둥 4개)**으로 정확한 벽·기둥 가림(오클루전)을 적용했습니다.
- **시간정지 비산 스타일 (`render_frozen_splash.py` → `piece_frozen/out_v2/`):** Pexels `7578547`(240프레임, 8.0초) 거실 중앙에서 14프레임에 **팡!** 터진 뒤 1.25초 만에 완전 시간정지되며(비정형 980개 방울, 3.6mm~98mm), **3개 투명 3D 벽/문틀 홀드아웃**과 동일한 **Pass-3 카메라 광학 피니시**를 적용했습니다.
- **인물 중심 시간정지 비산 (`render_frozen_human.py` → `piece_human/out/`):** Pexels `7251765`(225프레임, 25fps / 9.0초, 현대 지하철 승강장 3D 짐벌 아크 샷)에 **MediaPipe Pose 33관절 + 인물 세그멘테이션 마스크 + COLMAP 3D + 좌측 원형 콘크리트 기둥 3D 홀드아웃**을 결합해, 사람 몸통 주변에서 20프레임에 **팡!** 1,064개 비정형 방울이 사방으로 퍼져 시간정지되며 사람 앞·뒤 가림과 인물 초점 고정 심도 보케가 자연스럽게 작동합니다.

## 소스

- **Gallery (`piece_glass`):** Pexels `7578546` "Video of a House Interior", 촬영 Kindel Media, [Pexels License](https://www.pexels.com/license/).
  페이지: https://www.pexels.com/video/video-of-a-house-interior-7578546/
  (1920×1080, 30fps, 20번부터 270프레임 / 9.0초)
- **Frozen Splash (`piece_frozen`):** Pexels `7578547`, 촬영 Kindel Media, [Pexels License](https://www.pexels.com/license/).
  (1920×1080, 30fps, 120번부터 240프레임 / 8.0초, 6.11m 직진 샷)
- **Human Frozen Splash (`piece_human`):** Pexels `7251765` "A Man Waiting for a Train to Arrive", 촬영 MART PRODUCTION, [Pexels License](https://www.pexels.com/license/).
  페이지: https://www.pexels.com/video/a-man-waiting-for-a-train-to-arrive-7251765/
  (1920×1080, 25fps, 0번부터 225프레임 / 9.0초)
- 영상과 캐시는 git에 올리지 않습니다(`examples/liquid_glass/footage/`, `*/work/`, `*/out*/`).

## 실행

저장소 루트에서 `.venv` 파이썬으로 실행합니다.

```sh
# 1) 기본 / v2 + v7 브레이크다운
P=examples/liquid_glass/piece_glass
PY examples/shader_live/pipeline/slv.py prep   $P
PY examples/shader_live/pipeline/slv.py solve  $P
PY examples/shader_live/pipeline/slv.py plane  $P
PY examples/liquid_glass/render_glass.py       $P --still 55,75,85,135         # 스틸 검수 -> work/still_v2_sheet.jpg
PY examples/liquid_glass/render_glass.py       $P --passes --out out_v2        # 전체 렌더 + 브레이크다운 패스
PY examples/liquid_glass/breakdown_glass.py    $P --out out_v2 --lms           # v7 브레이크다운 + LMS 엔딩

# 2) 고밀도 리퀴드 글래스 갤러리 (5 군집 + 4 플루티드 + 188 위성 방울)
PY examples/liquid_glass/render_glass_dense.py $P --still 25,85,155,235        # 스틸 검수 -> work/still_dense_sheet.jpg
PY examples/liquid_glass/render_glass_dense.py $P --out out_dense              # 270프레임 렌더

# 3) 시간정지 비산 스타일 (팡! 비산 -> 980개 비정형 방울 공중 정지)
PF=examples/liquid_glass/piece_frozen
PY examples/liquid_glass/render_frozen_splash.py $PF --still 8,22,58,145,215   # 스틸 검수 -> work/still_frozen_v2_sheet.jpg
PY examples/liquid_glass/render_frozen_splash.py $PF --out out_v2              # 240프레임 렌더

# 4) 인물 중심 시간정지 비산 (MediaPipe Pose+마스크 + COLMAP 3D + 1,064개 비정형 방울)
PH=examples/liquid_glass/piece_human
PY examples/shader_live/pipeline/slv.py prep   $PH
PY examples/space_distortion/track_pose.py     $PH                             # MediaPipe Pose 33관절 + 인물 마스크
PY examples/shader_live/pipeline/slv.py solve  $PH                             # 인물 마스킹 COLMAP 3D
PY examples/shader_live/pipeline/slv.py plane  $PH
PY examples/space_distortion/trails.py         $PH                             # 33관절 3D 리프팅 -> work/limbs.npz
PY examples/liquid_glass/render_frozen_human.py $PH --still 10,28,60,135,210   # 스틸 검수 -> work/still_human_sheet.jpg
PY examples/liquid_glass/render_frozen_human.py $PH --out out                  # 225프레임 렌더
```

## 광학 구현

1. **이중 계면 스넬 굴절 (Entry + Internal March + Exit):** 1차 광선이 유리 SDF 표면에 닿으면 굴절률(`ior = 1.46`)로 내부 굴절 후 음수 SDF 내부를 마칭해 두께와 출사점을 구하고, 다시 공기로 굴절시켜 실사 플레이트 프록시에 재투영합니다.
2. **크라운 글래스 미세 분산 & Beer-Lambert 흡수:** 과도한 무지개 대신 `dispersion = 0.018` 수준의 미세 색분산과 저철분(low-iron) 유리 흡수 계수로 투명감을 유지합니다.
3. **실사 기반 프레넬 반사 & 스페큘러:** 반사 광선으로 실제 실내 플레이트를 샘플링하고, 창문 햇빛·천장 매입등 하이라이트와 메니스커스 림 글로우를 더합니다.
4. **3D 바닥 카우스틱 & 접촉 그림자:** `z = 0` 평면에서 태양 방향(`uSun`)으로 광선을 쏴 유리를 통과할 때 플루티드 평행 카우스틱 띠와 유체 굴절 카우스틱 웹을 바닥에 투사합니다.
5. **투명 3D 건축 홀드아웃(벽·기둥 오클루전):** 실사 벽·기둥의 수직 모서리를 COLMAP 카메라 포즈로 삼각측량해 3D AABB 홀드아웃(`piece_glass` 6개, `piece_frozen` 3개)을 배치하고, 레이마칭 최대 거리와 Pass-2 뎁스 버퍼를 `min(tGlass, tOcc)`로 제한해 벽·기둥 뒤 유리와 카우스틱을 픽셀 단위로 정확히 가립니다.
6. **Pass-3 카메라 광학 피니시 (`POST_FS`):** CG 영역에만 물리 얇은 렌즈 착란원(`CoC`) 기반 심도 렌즈블러(초점면 `0.68~0.72 px` OLPF 소프트니스 + 근경 스쳐 지나갈 때 최대 `10.5~11.0 px` 보케 블러, 기둥 뒤 블러 누출 차단), 미세 하이라이트 할레이션, 외곽 방사 색수차, 플레이트 측정 블랙 플로어(`0.016~0.017`) 및 프레임별 애니메이션 센서 그레인을 적용합니다(배경 실사 픽셀 무손실 유지).

## 확인한 것 (2026-11-20, Windows · RTX 4070 Ti SUPER)

| 단계 | 결과 |
| --- | --- |
| COLMAP (`pycolmap`) | 270/270 프레임 등록, 3D 점 6,192개, 절반 해상도 평균 재투영 오차 `0.668 px`, 보정 초점거리 `850.1 px` |
| 바닥 평면 (`frame 135`) | 957/2717 인라이어, 카메라 높이 `0.58 m → 2.12 m`, 이동 거리 `13.38 m` (`cam_h = 1.35 m` 가정 기준) |
| 렌더 (`out_v2`) | ModernGL OpenGL 3.30, 2×2 서브픽셀 AA, 1920×1080 270프레임 |
| 고밀도 갤러리 v2 (`out_dense_v2`) | 5 군집 + 4 플루티드 + 188 위성 방울 + 6개 3D 벽/기둥 홀드아웃 + Pass-3 광학 피니시, 270프레임 59.7초 렌더 |
| 시간정지 비산 v2 (`piece_frozen/out_v2`) | Pexels `7578547` COLMAP (240/240 프레임, 3D 점 4,931개, 재투영 `0.685 px`, `f = 741.8 px`, 이동 `6.11 m`), 980개 비정형 방울 + 3개 3D 벽/문틀 홀드아웃 + Pass-3 광학 피니시, 240프레임 50.5초 렌더 |
| 인물 중심 시간정지 (`piece_human/out`) | Pexels `7251765` MediaPipe Pose+세그멘테이션 225/225프레임 + COLMAP (225/225 등록, 3D 점 1,510개, 재투영 `0.632 px`, `f = 1615.0 px`, 인물 거리 `4.13~4.21 m`, 코 높이 `1.62 m`), 1,064개 비정형 방울 + 원형 콘크리트 기둥 3D 홀드아웃 + 인물 초점 고정 보케/그레인, 225프레임 52.4초 렌더 |

## 가정과 한계

- 카메라 높이(`piece_glass` 135프레임 `1.35 m`, `piece_frozen` 120프레임 `1.45 m`), 렌즈 왜곡 0, 태양/실내 광원 방향, 배경 프록시 거리(`7.0 m` / `6.2 m`)는 가정값입니다.
- 주요 벽·구조 기둥·문틀은 직육면체(AABB) 3D 홀드아웃으로 가림 처리했으며, 계단 난간·가구(의자/테이블) 등 복잡한 비정형 실사 프롭의 3D 메쉬 오클루전은 포함하지 않았습니다.
- 근경 페이드(`nearFade`) 없이 카메라 핀홀 여유거리를 확보해, 근경 오브젝트와 방울이 디졸브 없이 자연스러운 근경 렌즈 보케와 함께 프레임 밖으로 통과합니다.
- `out_dense_v2/`와 `piece_frozen/out_v2/`의 최종 인코딩 `Compare_1080x1350.mp4`는 전 구간 12프레임 검증 시트(`work/verify_dense_v2_compare.jpg`, `work/verify_frozen_v2_compare.jpg`)와 1:1 확대 크롭(`work/crop_pillar_check.jpg`, `work/crop_frozen_v2_check.jpg`)으로 시각 확인했으며, 실제 미디어 플레이어 실시간 재생 검수는 수행하지 않았습니다.
