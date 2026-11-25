# 3D Liquid Chrome Sculpting · 수은 리본 궤적 & 중력 낙하 바닥 웅덩이

MediaPipe 3D 포즈와 바닥 평면 3D 좌표(`P[N, 33, 3]`) 위에서 **양손 움직임을 따라 공중에 응고되는 3D 수은(리퀴드 크롬) 리본**과 **실제 중력(`g = -9.8 m/s²`)으로 떨어져 바닥 타일에 고이는 수은 웅덩이**를 네이티브 OpenGL(ModernGL, GLSL 3.30) 레이마칭으로 합성한 테스트 예제입니다.

## 소스

- Pexels `32109492` "Energetic Solo Dance in Urban Setting", 촬영 Anh Nguyen, [Pexels License](https://www.pexels.com/license/).
  페이지: https://www.pexels.com/video/energetic-solo-dance-in-urban-setting-32109492/
- 1920×1080, 25fps, 240프레임(9.6초). 모던 경사 유리 파사드 앞 타일 광장, 전신 일상 복장 솔로 댄서.
- 영상과 캐시는 git에 올리지 않습니다(`examples/liquid_chrome/footage/`, `*/work/`, `*/out/`).

## 실행

```sh
P=examples/liquid_chrome/piece_chrome
PY examples/shader_live/pipeline/slv.py prep   $P
PY examples/space_distortion/track_pose.py     $P
PY examples/shader_live/pipeline/slv.py solve  $P
PY examples/shader_live/pipeline/slv.py plane  $P
PY examples/space_distortion/trails.py         $P
PY examples/liquid_chrome/render_chrome.py     $P --still 55,115,190
PY examples/liquid_chrome/render_chrome.py     $P
```

## 구현 요약

1. **양손 3D 수은 리본 (`smin` SDF):** 최근 20프레임 양손 3D 궤적을 속도에 비례해 굵어졌다 가늘어지는 부드러운 smooth-min 캡슐로 연결합니다.
2. **중력 낙하 방울 & 바닥 웅덩이:** 손 속도가 임계치(`1.85 m/s`)를 넘을 때 실제 탄도 중력으로 낙하하는 수은 방울을 생성하고, `z = 0` 콘크리트 타일에 닿으면 인접 웅덩이와 매끄럽게 합쳐지는 볼록 메니스커스 풀을 만듭니다(뒤편 난간 앞 타일 영역으로 제한).
3. **360° 구면 거울 반사 & 가림 처리:** 실사 건축 파사드 플레이트를 360° 구면으로 한 번 매핑한 반사 + 지평선 대비 돔 + 태양 스페큘러 + `z = 0` 소프트 그림자 + MediaPipe 마스크 기반 깊이 인식 앞/뒤 가림을 적용합니다.

## 확인한 것 (2026-11-20, Windows · RTX 4070 Ti SUPER)

| 단계 | 결과 |
| --- | --- |
| MediaPipe Pose | 239/240 프레임 검출(1프레임 보간) |
| 카메라 / 축척 | 바닥 호모그래피 인라이어 최소 505 · 중앙값 581, `cam_h = 1.28 m` 가정 → 코 높이 중앙값 `1.61 m`, 댄서 거리 `4.58–6.21 m` |

## 가정과 한계

- 가로 화각(`60°`), 카메라 높이(`1.28 m`), 수평 틸트(`90°`), 태양 방향(`az -125°, el 52°`)은 가정값입니다.
- 환경 반사는 실사 플레이트와 합성 스튜디오 돔을 혼합한 근사이며 실제 현장 360° HDRI 캡처가 아닙니다.
- 테스트 단계이므로 브레이크다운 영상은 생성하지 않았습니다.
