# 3D LiDAR Spatial Scan · 실사 3D 구면 스캔 & 고도 등고선 웨이브

COLMAP으로 복원한 실사 3D 카메라 경로·바닥 평면(`z = 0`)과 **실제 4,198개 3D 건축 랜드마크**를 결합해, 카메라에서 퍼져 나가는 **3D 구면 LiDAR 스캔 펄스**를 네이티브 OpenGL(ModernGL, GLSL 3.30)로 합성한 테스트 예제입니다.

## 소스

- Pexels `7578547`, 촬영 Kindel Media, [Pexels License](https://www.pexels.com/license/).
  페이지: https://www.pexels.com/video/7578547/
- 1920×1080, 30fps, 원본 21.2초 중 120번부터 240프레임(8.0초). 주방 아일랜드·식탁·복도를 관통하는 짐벌 전진 샷.
- 영상과 캐시는 git에 올리지 않습니다(`examples/lidar_scan/footage/`, `*/work/`, `*/out/`).

## 실행

```sh
P=examples/lidar_scan/piece_lidar
PY examples/shader_live/pipeline/slv.py prep  $P
PY examples/shader_live/pipeline/slv.py solve $P
PY examples/shader_live/pipeline/slv.py plane $P
PY examples/lidar_scan/render_lidar.py       $P --still 30,115,195
PY examples/lidar_scan/render_lidar.py       $P
```

## 구현 요약

1. **실측 3D 깊이장 융합:** 통계적 이상치 제거를 거친 실제 COLMAP 3D 점군(4,198점)을 카메라 시점으로 투영해 중앙값 k-NN 깊이맵을 만들고, 바닥(`z = 0`)·천장(`z = 3.25 m`) 평면과 미디언·양방향 필터로 매끄럽게 융합합니다.
2. **3D 구면 스캔 웨이브 & 등고선:** 2.6초 간격으로 방출되는 3D 구면 파면이 지나가는 3D 표면 위에 `0.50 m` 공간 격자, `0.22 m` 수직 고도 등고선(`Z`), 건축 에지 와이어프레임을 모노크롬/아이스화이트 톤으로 드러냅니다.
3. **네이티브 `GL_POINTS` 점군 발광:** 실제 COLMAP 3D 랜드마크를 OpenGL `GL_POINTS`로 직접 투영해 구면 파면이 스칠 때 점군 노드가 점등됩니다.

## 확인한 것 (2026-11-20, Windows · RTX 4070 Ti SUPER)

| 단계 | 결과 |
| --- | --- |
| COLMAP (`pycolmap`) | 240/240 프레임 등록, 3D 점 4,931개(이상치 제거 후 4,198개), 절반 해상도 평균 재투영 오차 `0.685 px`, 보정 초점거리 `741.8 px` |
| 바닥 평면 | 199/1276 인라이어, 카메라 높이 `1.42 m → 1.41 m`, 이동 `6.11 m` (`cam_h = 1.45 m` 가정 기준) |
| 출력 | `LiDAR_Scan_Gallery_Composite.mp4`, `LiDAR_Scan_Gallery_Compare_1080x1350.mp4`, 12프레임 인코딩 시트 시각 검수 완료 |

## 가정과 한계

- 카메라 높이(`1.45 m`), 천장 높이(`3.25 m`), 렌즈 왜곡 0은 가정값입니다.
- 특징점이 희박한 흰 벽면 구간은 주변 랜드마크와 바닥·천장 프록시로 보간되므로 실제 벽면과 미세한 거리 오차가 있을 수 있습니다.
- 테스트 단계이므로 브레이크다운 영상은 생성하지 않았습니다.
