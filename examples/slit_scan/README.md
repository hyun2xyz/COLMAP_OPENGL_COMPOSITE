# Slit-Scan · 몸에서 흘러나오는 시간 줄무늬

고전 슬릿스캔(스트립 사진, Trumbull의 『2001』 복도, TouchDesigner 3D slitscan 튜토리얼)처럼 **슬릿 하나를 시간 위에 펼쳐** 줄무늬로 늘입니다.
슬릿은 춤추는 사람의 실루엣을 따라가는 선입니다. 슬릿 왼쪽은 원본 그대로이고, 오른쪽은 슬릿에 걸린 것이 시간 순서대로 길게 끌려 나갑니다.
그래서 몸은 오른쪽으로 길게 늘어난 줄무늬를 남기고, 가만히 있는 벽은 가로 줄무늬가 됩니다. 색은 더하지 않습니다. 모든 화소는 어떤 순간의 원본 색입니다.
네이티브 OpenGL(ModernGL, GLSL 3.30)입니다.

## 소스

- Pexels 8928132 "A Woman Dancing on the Sidewalk", 촬영 RDNE Stock project, [Pexels License](https://www.pexels.com/license/).
  페이지: https://www.pexels.com/video/a-woman-dancing-on-the-sidewalk-8928132/
- 1920×1080, 30fps, 원본 19초 중 30번부터 300프레임(10초). 밝은 외벽 앞 보도, 거의 고정 카메라.
- 영상은 저장소에 넣지 않습니다. `python scripts/fetch_assets.py footage 8928132`로 받습니다.

## 실행

저장소 루트에서. 포즈·카메라·관절은 [space_distortion](../space_distortion/README.md)의 도구를 그대로 씁니다.

```sh
P=examples/slit_scan/piece
PY examples/shader_live/pipeline/slv.py prep  $P      # 프레임 추출
PY examples/space_distortion/track_pose.py   $P      # MediaPipe 포즈 + 사람 마스크
PY examples/shader_live/pipeline/slv.py solve $P      # 카메라 (고정 카메라라 바닥 호모그래피)
PY examples/shader_live/pipeline/slv.py plane $P      # 바닥 평면
PY examples/space_distortion/trails.py       $P      # 관절을 장면 3D로 (코 높이 ~1.6m 확인; 사람까지의 거리로 m→px 환산)
PY examples/slit_scan/render_slit.py         $P --still 80,150,230   # 스틸: 원본 / 시간 지도 + 슬릿 선 / 결과
PY examples/slit_scan/render_slit.py         $P      # 전체 렌더 → out/*_Composite.mp4, *_Compare_1080x1350.mp4
```

브레이크다운은 모양이 확정된 뒤 `PY examples/space_distortion/breakdown_rules.py $P`로 만듭니다(`piece.json`의 `breakdown.steps`로 단계를 정함).

## 어떻게 만드는가

1. **링 버퍼.** 지난 `slit.K`(64) 프레임을 GPU 텍스처 배열에 돌려 담습니다.
2. **슬릿 선.** 행마다 사람 마스크의 가장 바깥 화소(`side` 쪽)에서 `offset_m`(사람 깊이에서 px로 환산; 음수면 몸 안쪽)만큼 떨어진 x.
   몸이 없는 행은 위아래에서 이어 붙이고, 행 방향으로 부드럽게 하고, 프레임 사이에서도 `smooth`로 흔들림을 줄입니다.
3. **펼치기.** 슬릿에서 거리 d만큼 오른쪽 화소는 **슬릿 열**의 `d × rate` 프레임 전 모습을 보입니다(이웃 두 프레임 사이 보간).
   슬릿 경계는 `feather_px`로 부드럽게 잇고, `rows: "body"`면 몸 근처 행만 펼칩니다.

## piece.json `slit` 값

| 키 | 뜻 |
| --- | --- |
| `K` | 링 버퍼 프레임 수(가장 먼 과거) |
| `side` | 펼치는 쪽(+1 오른쪽, −1 왼쪽) |
| `offset_m` | 몸 바깥 화소에서 슬릿까지(m, 음수 = 몸 안쪽) |
| `rate` | 1px당 과거 프레임 수(작을수록 줄무늬가 매끈하고 시간 폭이 짧음) |
| `feather_px`, `rows`, `smooth` | 슬릿 경계 부드럽게, 펼칠 행("all"/"body"), 슬릿 선의 시간 평활 |

## 확인한 것 (2026-11-19, Windows · RTX 4070 Ti SUPER)

| 단계 | 결과 |
| --- | --- |
| MediaPipe | 300/300 프레임 검출 |
| 카메라 | 호모그래피 풀이 인라이어 최소 877·중앙값 1290, 이동 0.04m(거의 고정) |
| 축척 | 가로 화각 60°·카메라 높이 0.93m 가정 → 코 높이 1.60m, 사람까지 3.5~5.0m |

## 한계

- 30fps라 빠른 동작에서는 프레임 사이 계단(약 29px 폭)이 보입니다. 프레임 사이는 섞기만 하고 움직임 보간은 하지 않습니다.
- 처음 `K` 프레임은 과거가 모자라 가장 오래된 프레임으로 채웁니다.
