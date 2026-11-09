# 포크로 시작하고 업데이트 받기

이 수업은 각자 GitHub 계정에 이 저장소를 **포크**해서 시작합니다. 내 포크에 실습 결과와 새 스크립트를
쌓고, 강사가 원본을 고치면 **pull**로 받아 이어 갑니다. 공통강의의 Git·GitHub CLI 세팅
(`gh auth login`)을 마친 컴퓨터를 기준으로 합니다.

| 이름 | 가리키는 곳 | 하는 일 |
| --- | --- | --- |
| `origin` | 내 포크 `내계정/COLMAP_OPENGL_COMPOSITE` | 내 작업을 push |
| `upstream` | 강사 원본 `hyun2xyz/COLMAP_OPENGL_COMPOSITE` | 업데이트를 받기만 함 |

## 1. 처음 한 번: 포크해서 받기

Codex나 Claude Code를 열고 붙여 넣습니다.

> https://github.com/hyun2xyz/COLMAP_OPENGL_COMPOSITE 를 제 GitHub 계정으로 포크해서 문서(Documents) 폴더에 받아 주세요.
> 공통강의에서 설정한 gh 인증을 쓰고, 같은 이름의 폴더가 이미 있으면 덮어쓰지 말고 알려 주세요.
> 끝나면 origin과 upstream 주소를 보여 주세요.

에이전트가 실행하는 명령은 다음과 같습니다. `gh`가 원본을 `upstream`으로 함께 등록합니다.

```sh
cd ~/Documents
gh repo fork hyun2xyz/COLMAP_OPENGL_COMPOSITE --clone
git -C COLMAP_OPENGL_COMPOSITE remote -v
```

```powershell
Set-Location "$env:USERPROFILE\Documents"
gh repo fork hyun2xyz/COLMAP_OPENGL_COMPOSITE --clone
git -C COLMAP_OPENGL_COMPOSITE remote -v
```

받은 뒤에는 **`COLMAP_OPENGL_COMPOSITE` 폴더를 Codex에서 다시 엽니다.** Git 폴더라 Codex가 쓰기 가능한 기본 모드로
열고, 이 저장소의 `AGENTS.md`와 `.agents/skills`를 자동으로 읽습니다. 그다음
[START-HERE](../START-HERE.md)의 문장을 붙여 넣습니다.

Codex 기본 모드에서는 인터넷을 쓰는 명령(git·gh·pip·npm·brew·winget)과 이 폴더 밖에 쓰는 명령마다
승인 창이 뜹니다. 무엇을 하는 명령인지 읽고 승인합니다. 보안 설정 자체를 끄자는 요청은 거절합니다.

포크는 **공개 저장소**입니다. 영상·프레임·렌더·로고 파일과 내 컴퓨터 경로(`tools.local.json`)는
`.gitignore`된 `work/` 등에만 두고 올리지 않습니다.

## 2. 내 작업 올리기

> 오늘 바뀐 파일을 보여 주고, 코드·문서 변경만 커밋해서 내 포크(origin)에 push해 주세요.
> 영상·렌더·개인 경로가 섞여 있으면 멈추고 알려 주세요.

```sh
git status
git add <올릴 파일>
git commit -m "무엇을 바꿨는지"
git push origin main
```

강사 원본과 충돌을 줄이려면 내 기능은 새 파일로 만들고, 원래 있던 파일은 필요한 줄만 고칩니다.

## 3. 강사 업데이트 받기

> 강사 원본(upstream)의 업데이트를 받아 주세요. 제 변경을 지우지 말고 병합하고, 충돌이 나면
> 양쪽 내용을 보여 주고 제게 물어봐 주세요. 끝나면 내 포크에도 push해 주세요.

```sh
git status
git fetch upstream
git merge upstream/main
git push origin main
```

- 커밋하지 않은 변경이 있으면 먼저 커밋한 뒤 병합합니다.
- 충돌이 나면 양쪽 내용을 모두 살려 고친 뒤 `git add` → `git commit`으로 병합을 마칩니다.
- `git reset --hard`, `git push --force`, GitHub 웹 **Sync fork**의 **Discard commits**는 쓰지 않습니다.
  내 작업이 지워집니다. Sync fork 버튼은 내 포크에 새 커밋이 없을 때만 씁니다.
- `upstream`이 없다고 나오면 한 번 등록합니다:
  `git remote add upstream https://github.com/hyun2xyz/COLMAP_OPENGL_COMPOSITE.git`

## 4. Git 세팅 전이라면

Git·gh가 없으면 먼저 공통강의의 Git 세팅을 합니다. ZIP으로 받은 폴더는 Git 폴더가 아니어서 Codex가
읽기 전용으로 열고, 강사 업데이트도 pull로 받을 수 없습니다.
