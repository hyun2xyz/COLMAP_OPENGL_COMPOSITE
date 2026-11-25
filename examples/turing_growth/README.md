# 3D World-Locked Turing Growth · 실사 광장 리액션-디퓨전 침식

상승하는 탑다운 드론 영상의 지면 평면(`z = 0`)에 **GPU 실시간 1024×1024 Gray-Scott Reaction-Diffusion(튜링 패턴) PDE**를 월드락으로 결합해, 원형 분수대 테두리에서 유기적 산호/석회질 부조가 실제 광장 바닥 타일을 따라 번져 나가는 테스트 예제입니다. 네이티브 OpenGL(ModernGL, GLSL 3.30) 핑퐁 FBO로 연산합니다.

## 소스

- Pexels `14012992` "Park and Streets in Town", 촬영 SAULO LEITE, [Pexels License](https://www.pexels.com/license/).
  페이지: https://www.pexels.com/video/park-and-streets-in-town-14012992/
- 1920×1080, 23.976fps, 216프레임(9.0초). 원형 분수대와 테라코타 석재 광장을 수직 상승하며 촬영한 드론 샷.
- 영상과 캐시는 git에 올리지 않습니다(`examples/turing_growth/footage/`, `*/work/`, `*/out/`).

## 실행

```sh
P=examples/turing_growth/piece_turing
PY examples/shader_live/pipeline/slv.py prep   $P
PY examples/shader_live/pipeline/slv.py solve  $P
PY examples/shader_live/pipeline/slv.py plane  $P
PY examples/turing_growth/render_turing.py     $P --still 15,105,205
PY examples/turing_growth/render_turing.py     $P
```

## 구현 요약

1. **월드 공간 GPU Gray-Scott PDE:** `1024×1024` 부동소수점 핑퐁 텍스처를 실제 광장 `[-26 m, +26 m]²` 평면에 1:1 매핑해 프레임당 28스텝씩 확산·반응을 진행합니다(`F = 0.0365, K = 0.0615`).
2. **촉각적 3D 노멀맵 부조 & 햇빛 음영:** 농도 `B`에서 월드 공간 3D 노멀과 태양 방향 미세 그림자를 계산해, 석재가 실제로 솟아올라 젖은 광택과 햇빛 하이라이트를 받는 것처럼 합성합니다.
3. **자동 식생·분수대 마스킹:** 초록 수목 캐노피와 중앙 원형 분수대 수조(`r < 5.45 m`)를 제외해 부조가 실제 테라코타 보도블록 위로만 기어 올라갑니다.

## 확인한 것 (2026-11-20, Windows · RTX 4070 Ti SUPER)

| 단계 | 결과 |
| --- | --- |
| 카메라 풀이 | 평면 호모그래피(`tilt 0°`), 인라이어 최소 1,370 · 중앙값 2,377 |
| 평면 정합 | 1,555/1,555 인라이어, 카메라 높이 `28.00 m → 53.14 m`, 수직 상승 `25.42 m` (`cam_h = 28 m` 가정 기준) |
| 출력 | `Turing_Growth_Plaza_Composite.mp4`, `Turing_Growth_Plaza_Compare_1080x1350.mp4`, 12프레임 인코딩 시트 시각 검수 완료 |

## 가정과 한계

- 초기 드론 고도(`28 m`), 가로 화각(`72°`), 수직 하향(`tilt 0°`), 태양 방향(`az -42°, el 48°`)은 가정값입니다.
- 식생/분수대 분리는 색상·반경 기반 마스킹이므로 광장 위 아주 작은 보행자 그림자는 일부 부조와 겹칠 수 있습니다.
- 테스트 단계이므로 브레이크다운 영상은 생성하지 않았습니다.
