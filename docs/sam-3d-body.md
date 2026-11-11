# SAM 3D Body: 쓰기 전에 등록부터

이 저장소의 몸 추적은 **MediaPipe로 바로 시작합니다.** SAM 3D Body는 더 정밀한 선택 사항이고, 쓰려면 먼저
접근 신청을 해 두어야 합니다. 승인이 나기 전까지는 MediaPipe로 진행합니다.

## 무엇이 다른가

| | MediaPipe Pose (기본) | SAM 3D Body (선택) |
| --- | --- | --- |
| 결과 | 관절 33개 | 전신 메시(정점 18,439개)와 키포인트 70개, 발끝·발뒤꿈치 포함 |
| 받기 | 바로 받을 수 있음 (약 30MB) | Hugging Face에서 접근 신청 뒤 승인 필요 (체크포인트 폴더 약 2.7GB) |
| 필요한 것 | CPU로도 동작 | PyTorch와 성능 좋은 그래픽 카드, 별도 Python 환경 |
| 라이선스 | Apache-2.0 | SAM License |
| 이 저장소 | `examples/fruit_physics`가 사용 | 코드와 가중치 모두 들어 있지 않음 |

논문 정리본의 방법 B에서 몸의 거리 추정이 더 안정적이었던 쪽이 SAM 3D Body입니다. 그래도 과일이 몸에 부딪치는
예제 자체는 MediaPipe만으로 끝까지 만들 수 있습니다.

## 지금 해 둘 일: 접근 신청

1. [Hugging Face](https://huggingface.co/join) 계정을 만듭니다.
2. 모델 페이지를 엽니다: https://huggingface.co/facebook/sam-3d-body-dinov3
3. 페이지의 신청 양식을 **직접** 작성해 제출합니다. 이름, 생년월일, 국가, 소속을 묻고 SAM License 동의를 받습니다.
   에이전트에게 대신 작성시키지 않습니다.
4. 승인을 기다립니다. 승인 여부와 걸리는 시간은 Meta가 정합니다. 일부 지역은 거절될 수 있다고 안내돼 있습니다.

신청한 날에 바로 쓸 수 없을 수 있으므로, 수업 전에 미리 해 두는 것이 좋습니다.

## 승인된 뒤

공식 저장소의 설치 문서를 따릅니다. 이 저장소의 `.venv`와는 별도의 환경을 만듭니다.

- 코드와 설치 문서: https://github.com/facebookresearch/sam-3d-body ([INSTALL.md](https://github.com/facebookresearch/sam-3d-body/blob/main/INSTALL.md))
- PyTorch 설치: https://pytorch.org/get-started/locally/
- 체크포인트 받기 (승인된 계정으로 로그인한 뒤):

```sh
hf auth login
hf download facebook/sam-3d-body-dinov3 --local-dir checkpoints/sam-3d-body-dinov3
```

- 논문: X. Yang 외, "SAM 3D Body: Robust Full-Body Human Mesh Recovery", arXiv:2602.15989, 2026.

SAM 3D Body의 결과를 이 저장소의 과일 예제에 연결하는 스크립트는 아직 들어 있지 않습니다. 강의에서 다룰 때 추가합니다.

## 에이전트에게

- 학생이 "승인됐다"고 말하기 전에는 SAM 3D Body 설치를 시작하지 않습니다. MediaPipe 경로로 진행합니다.
- 접근 신청 양식, 라이선스 동의, Hugging Face 로그인은 학생이 직접 합니다.
- 가중치를 다른 곳에서 찾아 받지 않습니다. 가중치와 토큰은 커밋하지 않습니다.
