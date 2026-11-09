# 재료 찾기 — 영상과 HDRI

학생이 "이런 장면에 이런 걸 합성하고 싶다"고 말하면, **에이전트가 이 문서대로 직접 재료를 찾습니다.** 학생에게 먼저 검색해 오라고 하지 않습니다. 후보를 고른 뒤에는 근거(페이지, 제작자, 라이선스, 맞는 이유, 본 프레임)를 보여 주고, 긴 카메라 계산을 시작합니다.

도구는 `scripts/find_materials.py`입니다. 저장소 폴더에서 프로젝트 `.venv`의 Python으로 실행합니다.

```sh
python -B scripts/find_materials.py footage "drone abandoned factory interior"
python -B scripts/find_materials.py hdri overcast forest
python -B scripts/find_materials.py sheet <다운로드 URL 또는 파일> --start 2 --duration 4
python -B scripts/find_materials.py new-shot my_shot --like hall_water \
  --page <영상 페이지> --download <다운로드 URL> --creator "<제작자>" --license pexels \
  --start 2 --duration 4 --hdri <Poly Haven id>
```

## 1. 어디에서 찾는가

| 배포처 | 라이선스 | 쓰는 법 |
| --- | --- | --- |
| [Pexels Videos](https://www.pexels.com/videos/) | [Pexels License](https://www.pexels.com/license/) — 무료 사용·수정 허용, 표기 선택, **CC0 아님** | `PEXELS_API_KEY`가 있으면 `footage`가 API로 검색합니다. 없으면 검색 페이지 주소를 출력합니다. |
| [Pixabay Videos](https://pixabay.com/videos/) | [Pixabay Content License](https://pixabay.com/service/license-summary/) — 무료 사용·수정 허용, **CC0 아님** | `PIXABAY_API_KEY`가 있으면 함께 검색합니다. |
| [Poly Haven HDRIs](https://polyhaven.com/hdris) | [CC0](https://polyhaven.com/license) | `hdri`가 공개 API로 검색합니다(키 필요 없음). API를 쓰므로 "Powered by Poly Haven"을 표기합니다. |

API 키는 각 사이트에서 무료로 받을 수 있습니다([Pexels](https://www.pexels.com/api/), [Pixabay](https://pixabay.com/api/docs/)). 키는 환경변수나 무시되는 `.env`에만 두고, 화면에 출력하거나 커밋하지 않습니다. 키가 없으면 에이전트가 브라우저로 검색 페이지를 엽니다. 브라우저도 없다면 학생에게 검색 페이지 주소를 알려 줍니다.

**쓰지 않는 곳**: 인스타그램·틱톡·유튜브 등 SNS 영상, 영화·드라마·뮤직비디오, 라이선스 표시가 없는 페이지, "무료"라고만 적힌 모음 사이트, AI 생성 영상(카메라가 물리적으로 일관되지 않습니다). 무료라고 해서 저작권이 없는 것은 아닙니다.

## 2. 영상 고르는 기준

카메라 복원은 **시차**(가까운 것과 먼 것이 다르게 움직이는 것)로 깊이를 풉니다. 아래 조건을 모두 만족해야 합니다.

- **카메라가 이동합니다.** 드론 비행, 걷기, 달리, 트래킹, 오빗 숏이 좋습니다. 제자리에서 고개만 돌리는 팬·틸트는 깊이가 풀리지 않습니다.
- **하드컷이 없습니다.** 쓸 4~5초 안에 컷, 디졸브, 속도 변화(스피드 램프), 타임랩스가 없어야 합니다.
- **줌이 고정입니다.** 줌인·줌아웃, 강한 손떨림 보정 왜곡, 화면 흔들림 효과가 없어야 합니다.
- **정적인 질감이 화면 대부분을 채웁니다.** 콘크리트, 벽, 바닥, 바위, 잔디처럼 움직이지 않는 질감이 좋습니다. 물·하늘·흔들리는 나뭇잎·사람·차가 화면 대부분인 숏은 피합니다.
- **선명합니다.** 모션 블러가 심하거나, 노이즈가 많거나, 노출이 깜빡이는 숏은 피합니다.
- **물체를 둘 평평한 자리가 보입니다.** 그 자리 앞을 기둥이나 나무가 지나가지 않아야 합니다. 이 패키지는 가림 마스크를 만들지 않습니다.
- **가로 영상**이고 1280×720 이상이며, 글자·워터마크·레터박스가 없습니다.

숏 종류별로 이렇게 찾습니다.

| `kind` | 맞는 장면 | 검색어 예 |
| --- | --- | --- |
| `water` (물 + 링) | 실내나 그늘진 곳, 바닥이 넓게 보이는 전진·측면 이동 | `drone abandoned building interior`, `walking empty warehouse`, `parking garage tracking` |
| `rigid_spheres` (공) | 위에서 내려다보는 넓은 바닥, 잔디·광장 | `drone top down courtyard`, `aerial plaza`, `top view lawn` |
| `cloth` (천) | 야외 평지, 바람이 어울리는 곳, 해 그림자가 뚜렷한 곳 | `aerial quarry`, `drone field slow`, `meadow flyover` |

검색어는 영어가 결과가 많습니다. `drone`, `aerial`, `tracking shot`, `walking`, `dolly`, `fly through`를 장소 이름과 함께 넣습니다.

## 3. 고르는 순서

1. **검색**: `footage "<검색어>"`로 후보를 받습니다. 6초 이상 가로 영상만 남깁니다.
2. **라이선스 확인**: 후보마다 배포처 페이지를 직접 열어 라이선스, 제작자 이름, 영상 제목을 확인합니다. 검색 결과만 보고 판단하지 않습니다. 사람 얼굴, 상표, 사유지가 두드러지면 피합니다.
3. **눈으로 보기**: `sheet <다운로드 URL> --start <초> --duration 4`로 쓸 구간의 프레임 12장을 한 장에 모아 봅니다. 위 기준을 하나씩 확인하고, 무엇을 봤는지 말합니다. 맞는 구간이 없으면 다음 후보로 넘어갑니다.
4. **HDRI 고르기**: `hdri <단어>`로 영상과 빛이 비슷한 HDRI를 찾습니다. 실내·야외, 해가 보이는지 흐린지, 시간대(아침·한낮·노을)를 맞춥니다. `--category indoor`처럼 분류로 좁힐 수도 있습니다. HDRI는 현장 조명을 측정한 것이 아니라 **비슷한 조명을 고른 가정값**입니다.
5. **학생에게 보여 주기**: 영상 페이지, 제작자, 라이선스, 쓸 구간, 고른 이유, 확인한 시트, HDRI를 짧게 보여 줍니다.
6. **숏 만들기**: `new-shot`이 영상을 `work/<숏>/source/`에 받습니다. 이어서 SHA-256을 계산하고, 길이·크기를 확인하고, 구간 검수 시트(`plate_sheet.jpg`)를 만든 뒤, 가장 비슷한 예제를 바탕으로 `shots/<숏>.json`을 씁니다. 같은 이름의 숏이 있으면 덮어쓰지 않습니다.
7. **카메라로 확인**: `python -m pipeline composite <숏>`을 실행합니다. `qa-camera`를 통과하지 못하면 [워크플로](workflow.md#4-qa-camera--통과해야-다음으로-3초)대로 초점이나 구간을 한 가지씩 바꿔 봅니다. 그래도 안 되면 억지로 맞추지 말고 다른 영상을 고릅니다.

## 4. 남기는 기록

`shots/<숏>.json`의 `source`와 `hdri`에 다음을 남깁니다. `new-shot`이 자동으로 채우고, 사람이 페이지와 대조합니다.

- 영상: 제목, 제작자, 페이지 주소, 라이선스 이름과 주소, 허용·금지 요약, 다운로드 주소, SHA-256, 확인한 날짜.
- HDRI: 파일 이름, 제목, 제작자, 페이지, CC0, "Powered by Poly Haven", 다운로드 주소, SHA-256.

`fetch`는 다른 컴퓨터에서도 같은 주소에서 받아 해시를 대조합니다. 배포처가 파일을 바꾸면 멈추므로, 라이선스를 다시 확인한 뒤 새로 고릅니다.

## 5. 지키는 것

- 원본 영상, 프레임, HDRI는 `work/`(또는 `VFX_WORK`)에만 둡니다. 저장소에 커밋하지 않습니다.
- Pexels·Pixabay 원본을 수정하지 않은 채 다시 배포하거나 판매하지 않습니다. 제작자나 배포처가 이 작업을 보증하거나 후원하는 것처럼 표시하지 않습니다.
- 결과물을 공개할 때는 의무가 아니더라도 제작자와 배포처를 표기하는 것을 권합니다(예: "Footage: Florian Delée / Pexels · HDRI: Poly Haven").
- 라이선스를 확인하지 못한 영상은 쓰지 않습니다. 확인한 것과 추정한 것을 구분해서 말합니다.
