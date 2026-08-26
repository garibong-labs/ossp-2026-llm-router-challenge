<!--
SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
SPDX-License-Identifier: Apache-2.0
-->

# 제출 컨테이너

[`Dockerfile`](Dockerfile)은 이 fork의 제출 정책인 safe-margin 라우터를 표준
`router-run` 인터페이스로 실행합니다. 라우터 실행 입력 JSON의 컨테이너 내부
경로는 `/challenge/input/inputs.json`, 선택 결과 JSON의 경로는
`/challenge/output/submission.json`, 임시 경로는 `/tmp`입니다.

[`entrypoint.py`](entrypoint.py)는 정책을 다시 쓰지 않고
[`../baselines/safe_margin.py`](../baselines/safe_margin.py)의 `main`을 그대로
호출합니다. 개발용 실행기와 제출 이미지가 같은 구현 하나를 사용하므로 정책이
갈라질 수 없습니다. 이미지에 넣는 실행 파일은 다음뿐입니다.

| 이미지 경로 | 용도 |
| --- | --- |
| `/opt/router/entrypoint.py` | 진입점, safe-margin `main`으로 위임 |
| `/opt/router/baselines/safe_margin.py` | 제출 라우팅 정책 |
| `/opt/router/baselines/hash_regex.py` | 특징 추출과 artifact 파서 |
| `/opt/router/baselines/hash-regex-public.v1.json` | 공개 hash-regex artifact |
| `/opt/router/ossp_router/` | 공개 프로토콜·정책 자원 모듈 |

공개 artifact는 라우터 모듈 옆에 함께 들어가며 `--artifact`의 기본값입니다.
운영자는 기존과 같이 `--input`, `--tier`, `--output`만 전달합니다. 공개
입력·outcome, 개발 도구, 테스트, 학습 자료와 빌드 산출물은
[`../.dockerignore`](../.dockerignore)에서 제외하며, 실제 포함 파일 목록은
[`../tests/test_container_entrypoint.py`](../tests/test_container_entrypoint.py)가
검사합니다.

구체적인 인자, 파일 권한, 제한 시간 초과와 비정상 종료, 출력 검증,
CPU, RAM, 프로세스·스레드 수의 최종 한도는
[`../docs/RUNTIME.md`](../docs/RUNTIME.md)에 정의합니다. 운영자 측 기술 장애,
최대 3회 실행, 첫 유효 결과와 전체 실격 사유는
[`../docs/ENFORCEMENT.md`](../docs/ENFORCEMENT.md)에 정의합니다.

컨테이너는 네트워크 없이, 비특권 UID/GID `65532:65532`, 읽기 전용 파일 시스템에서
실행하도록 설계했습니다. 참가자에게는 시도별 4 MiB 제한 출력 볼륨과
256 MiB `/tmp`만 쓰기 가능하며 GPU나 별도 device를 전달하지 않습니다.
공유 메모리는 제공하지 않고 이미지의 모든 `VOLUME` 선언은 실행 전에
거부합니다. 기반 이미지 출처와 다이제스트는
[`BASE_IMAGE.md`](BASE_IMAGE.md)에 기록합니다.

출력 회수, Docker 자원 정리와 장애 복구 방식은
[`../docs/OPERATIONS.md`](../docs/OPERATIONS.md)에 정의합니다.

## 제출과 무관한 실험 전용 이미지

이 디렉터리에는 제출 이미지 외에 두 개의 측정 전용 Dockerfile이 있습니다.
둘 다 제출·평가 경로에 들어가지 않으며 위의 `/opt/router` 파일 목록을 바꾸지
않습니다.

| 파일 | 용도 |
| --- | --- |
| [`measurement.Dockerfile`](measurement.Dockerfile) | baseline 런타임 벤치마크와 자원 한도 동결 |
| [`semantic-measurement.Dockerfile`](semantic-measurement.Dockerfile) | semantic upgrade-event 실험의 `linux/arm64` 추출 실행 가능성 측정 |

`semantic-measurement.Dockerfile`은 동결한 multilingual-e5-small artifact 6개와
고정 extraction 의존성만 담고, 빌드 마지막 단계에서 크기와 SHA-256을 다시
확인합니다. 이 이미지는 제출 라우터를 담지 않으며 상위
[`../.dockerignore`](../.dockerignore) 대신 자신의
[`semantic-measurement.Dockerfile.dockerignore`](semantic-measurement.Dockerfile.dockerignore)만
사용하므로 제출 이미지의 build context는 그대로입니다. 측정 절차와 관측값은
[`../baselines/README.md`](../baselines/README.md)에 있습니다.

Colima의 Docker 호환 실행기에서 실제 이미지 빌드와 네트워크 없음, GPU 없음,
비특권 사용자, 읽기 전용 루트 파일 시스템 조건을 검증했습니다. 통합 테스트는
`OSSP_RUN_CONTAINER_TESTS=1`로 켤 수 있습니다. 공개 Train/Dev 호스트·격리
컨테이너 측정 결과와 동결한 최종 자원 한도는
[`../docs/runtime-benchmark.md`](../docs/runtime-benchmark.md)에 있습니다.
측정과 한도 동결 절차는
[`../docs/APPLE_SILICON_MEASUREMENT.md`](../docs/APPLE_SILICON_MEASUREMENT.md)를
따릅니다.

참가자가 자신의 최종 이미지를 같은 공개 Train/Dev와 자원 제한으로 확인하는
명령은 [`../docs/RUNTIME.md`](../docs/RUNTIME.md#로컬-검증)에 안내합니다.
