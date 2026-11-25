# 3D Vision Quadtree HUD · 어댑티브 쿼드트리 & 3D 메트릭 블롭 트래킹

2026년 TouchDesigner POPs·`Blob Track`·Code Brutalism 트렌드를 실사 3D 파이프라인으로 옮긴 테스트 예제입니다.
MediaPipe 포즈(33관절·마스크)와 바닥 평면 3D 좌표(`P[N, 33, 3]`, 미터 단위), 몸 주변 게이트 옵티컬 플로우를 결합해 **네이티브 OpenGL(ModernGL, GLSL 3.30) 4단계 어댑티브 쿼드트리**와 **실제 3D 원근 바운딩 케이지·바닥 레티클·터렛 PIP**를 합성합니다.

## 소스

- Pexels `8928133`, 촬영 RDNE Stock project, [Pexels License](https://www.pexels.com/license/).
  페이지: https://www.pexels.com/video/8928133/
- 1920×1080, 30fps, 원본 15.07초 중 150번부터 252프레임(8.4초). 고정 카메라, 콘크리트 블록과 메탈 셔터 배경.
- 영상과 캐시는 git에 올리지 않습니다(`examples/vision_quadtree/footage/`, `*/work/`, `*/out/`).

## 실행

저장소 루트에서 `.venv` 파이썬으로 실행합니다. 포즈·3D 리프팅은 `space_distortion` 도구를 재사용합니다.

```sh
P=examples/vision_quadtree/piece_vision
PY examples/shader_live/pipeline/slv.py prep   $P
PY examples/space_distortion/track_pose.py     $P
PY examples/shader_live/pipeline/slv.py solve  $P
PY examples/shader_live/pipeline/slv.py plane  $P
PY examples/space_distortion/trails.py         $P
PY examples/vision_quadtree/render_vision.py   $P --still 35,125,215   # 스틸 검수 -> work/still_sheet.jpg
PY examples/vision_quadtree/render_vision.py   $P                      # 전체 렌더 -> out/*_Composite.mp4, *_Compare_1080x1350.mp4
```

## 구성 요소

1. **GPU 4단계 어댑티브 쿼드트리 (`120 → 60 → 30 → 15 px`):** `1920×1080`을 정확히 `16×9`로 나누는 매크로 격자에서 출발해, 움직임 에너지(몸 주변 게이트 Farneback 옵티컬 플로우 + 마스크 에지 + 3D 관절 속도)가 높은 셀만 재귀 분할합니다.
2. **Code Brutalism 셀 처리:** 얼굴 보호 반경 바깥의 활성 웨이크 셀에 5×5 비트맵 **글리프 디더링**과 **1.45× 광학 루페 확대**, 피크 속도 시그널 하이라이트(`#EBFF47`)를 선택적으로 적용합니다.
3. **3D 원근 바닥 레티클 & 발자국 궤적:** 댄서 발바닥의 실제 바닥 평면 좌표 `(fx, fy, 0)`에 `0.48 m / 0.92 m` 동심원과 최근 54프레임 3D 발 이동 궤적을 원근 투영합니다.
4. **3D 메트릭 바운딩 케이지 & 다중 노드 웹:** 3D 관절 범위로 만든 8꼭짓점 3D 박스(`H`, `D` 미터 표기)와 근접 관절 간 3D 거리 벡터, 우측 빈 셔터 공간에 히스테리시스 스무딩으로 타깃을 따라가는 모노크롬 PIP + 실시간 속도 스파크라인을 배치합니다.

## 확인한 것 (2026-11-20, Windows · RTX 4070 Ti SUPER)

| 단계 | 결과 |
| --- | --- |
| MediaPipe Pose | 252/252 프레임 검출 |
| 카메라 풀이 | 바닥 호모그래피, 인라이어 최소 2,172 · 중앙값 2,279, 이동 `0.00 m` (고정) |
| 축척 | 가로 화각 `58°` · 카메라 높이 `0.96 m` 가정 → 코 높이 중앙값 `1.60 m`, 댄서 거리 `4.27–4.93 m` |
| 출력 확인 | `Vision_Quadtree_Concrete_Composite.mp4` (22.0 MB), `Vision_Quadtree_Concrete_Compare_1080x1350.mp4` (14.5 MB), 인코딩 후 12프레임 시트 시각 검수 완료 |

## 가정과 한계

- 가로 화각(`58°`), 카메라 높이(`0.96 m`), 수평 틸트(`90°`)는 가정값입니다.
- 반복되는 메탈 셔터 패턴의 미세 압축 노이즈가 옵티컬 플로우에 잡히는 현상을 막기 위해, 플로우 에너지는 MediaPipe 사람 마스크 주변 웨이크 영역과 프레임 밝기 차분으로 게이팅했습니다.
- 테스트 단계이므로 브레이크다운 영상은 생성하지 않았습니다.
