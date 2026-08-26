<!--
SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
SPDX-License-Identifier: Apache-2.0
-->

# Baseline

README의 [Quickstart](../README.md#quickstart-baseline에서-시작하기)에서
실행과 검증 흐름을 먼저 확인한 뒤, 아래 예제 중 목적에 가까운 구현을 출발점으로
사용할 수 있습니다. 실제 제출에서는 `src/ossp_router/heuristic.py`를 바꾸거나
같은 `router-run` 인터페이스를 제공하는 구현으로 교체하면 됩니다.

## 모든 문항에 경량 모델 선택

[`always_light.py`](always_light.py)는 모든 문항에 `ax31-light`를 선택하고
세 등급 제출 파일을 한꺼번에 만듭니다. 점수와 비용 계산을 확인하기 위한
가장 단순한 baseline입니다.

## 약한 prompt-heuristic baseline

[`prompt_heuristic.py`](prompt_heuristic.py)는 한 번에 한 등급의 제출 파일을
만듭니다. 문항마다 prompt 또는 messages의 본문에서 다음 단순 특징을 직접
계산합니다.

- 문자·단어·문장 수와 메시지 수
- 한글 문자 비율
- 코드 형태와 수학 기호 수
- 숫자 밀도와 장문 문맥 여부
- 일부 일반적인 추론·분석 어휘

선택 점수는 길이, 코드, 수학, 숫자, 메시지 구조와 장문 여부에 고정된 작은
정수 가중치를 더해 계산합니다. Fast와 Balanced는 장문이 아닌 프롬프트 중
복잡도 임계값을 넘은 문항만 `ax31`로 보내고, Premium은 모든 문항에 `ax31`을
사용합니다. 학습된 출력 길이 예측 없이 K1 비용을 과소평가하지 않도록
`axk1-think`는 선택하지 않습니다. 이는 콘텐츠 기반 라우팅과 등급 전달 방식을
보여 주는 의도적으로 약한 예입니다.

이 라우터는 문항 ID, 입력 위치, 과제명, 출처, 모델 답변, 정답이나 평가
결과를 읽지 않습니다. 모델 선택 함수는 문항 내용과 실행 등급만 받습니다.
ID·순서 변경과 반복 실행의 결정성은
[`../tests/test_prompt_heuristic.py`](../tests/test_prompt_heuristic.py)에서
검사합니다.

```console
PYTHONPATH=src python3 baselines/prompt_heuristic.py \
  --input data/toy/inputs.json \
  --tier balanced \
  --output build/prompt-heuristic-balanced.json
```

실제 비용과 점수는 모델별 평가 결과가 있을 때 self-check로 확인합니다.
라우터 실행 시점에는 모델별 평가 결과를 실행 입력으로 전달하지 않습니다.

## 비용을 함께 배분하는 feature-budget baseline

[`feature_budget.py`](feature_budget.py)는 LM이나 학습 가중치 없이 위 특징에
형식 추론, 프로그램 분석, 다중 제약과 단순 변환 표지를 조금 더합니다. 각
문항을 독립 임계값으로만 처리하지 않고, 같은 특징 점수의 문항을 한 묶음으로
두어 등급별 추정 비용 안에서 전체 묶음을 승격합니다. 따라서 입력 순서나
문항 ID로 동률을 깨지 않습니다.

실제 모델별 생성 출력 토큰 수는 라우터에 제공되지 않으므로 비용 추정은 토큰
단가 비율과 프롬프트 길이에 기반한 대리값입니다. 예산의 85%만 사용해 여유를
둡니다. 학습된 출력 길이 예측 없이 K1 비용을 안전하게 추정하기 어려우므로 이
baseline은 `ax31-light`와 `ax31`만 배분합니다. 실제 Train/Dev의 `self-check`를
대체하지 않습니다.

```console
for tier in fast balanced premium; do
  PYTHONPATH=src python3 baselines/feature_budget.py \
    --input data/toy/inputs.json \
    --tier "$tier" \
    --output "build/feature-budget/$tier.json"
done
```

형식 확인용 toy 자료에서는 all-light가 `0.5`, `prompt-heuristic`과 이
baseline이 각각 `0.58`입니다.

## 학습형 hash-regex 선형 baseline

[`hash_regex.py`](hash_regex.py)는 명시적 정규식 특징과 단어 unigram·bigram의
signed feature hashing을 함께 사용합니다. [`train_hash_regex.py`](train_hash_regex.py)는
공개 Train의 모델별 score와 비용으로 여섯 개의 ridge 회귀 head를 학습합니다.
score와 log-cost를 각각 세 모델에 대해 예측하고, out-of-fold 예측에서 등급별
예산 안전계수를 고릅니다.

Premium에서는 비용 변동이 큰 K1 선택을 기본 안전계수로 먼저 확정합니다. 그
선택을 유지한 채, 전체 예측 비용이 Premium 한도의 65%를 넘지 않는 범위에서
예측 score가 개선되는 Light 선택만 AX31로 추가 승격합니다. Fast와 Balanced에는
이 추가 단계가 적용되지 않습니다.

학습에는 BSD-3-Clause 라이선스의 NumPy만 사용합니다. 생성된 JSON에는 전역
평균·스케일·회귀계수, 등급별 안전계수와 입력 파일 전체의 재현성 해시만
들어갑니다. prompt, 문항 ID, 문항별 특징이나 선택은 저장하지 않습니다.
실제 라우터는 표준 라이브러리만 사용하므로 NumPy를 제출 이미지에 넣을
필요가 없습니다.

공개 자료를 생성한 뒤 Train으로 회귀계수를 학습하고 Dev로 등급별 안전계수
세 값만 보정합니다.

```console
python3 -m pip install -r baselines/requirements-train.txt

PYTHONPATH=src python3 baselines/train_hash_regex.py \
  --input data/materialized/train/inputs.json \
  --outcomes data/train/outcomes.json \
  --validation-input data/materialized/dev/inputs.json \
  --validation-outcomes data/dev/outcomes.json \
  --artifact build/hash-regex/artifact.json \
  --report build/hash-regex/train-report.json

for tier in fast balanced premium; do
  PYTHONPATH=src python3 baselines/hash_regex.py \
    --input data/materialized/dev/inputs.json \
    --artifact build/hash-regex/artifact.json \
    --tier "$tier" \
    --output "build/hash-regex/dev-$tier.json"
done
```

같은 명령으로 만든 전역 계수 학습 파일
[`hash-regex-public.v1.json`](hash-regex-public.v1.json)을 함께 제공합니다.
따라서 학습 없이 아래처럼 바로 실행할 수도 있습니다.

```console
PYTHONPATH=src python3 baselines/hash_regex.py \
  --input data/materialized/dev/inputs.json \
  --artifact baselines/hash-regex-public.v1.json \
  --tier balanced \
  --output build/hash-regex/dev-balanced.json
```

## safe-margin MVP 라우터

[`safe_margin.py`](safe_margin.py)는 이 저장소의 첫 동작 MVP 정책입니다.
특징 추출과 예측은 공개 hash-regex 자료를 그대로 재사용하지만, 선택 정책은
한도에 근접하지 않는 보수적 hybrid로 새로 만들었습니다.

- **비용 추정을 구조적으로 보수화합니다.** 학습된 log-cost head 값을 공개
  정책의 입력 토큰 단가 비율(`ax31` 2.127배, `axk1-think` 6.565배)로 먼저
  바닥을 깔고, 모델별 여유 계수를 곱합니다. 학습 head가 새로운 프롬프트
  분포에서 흔들려도 승격 비용을 단가 비율보다 싸게 볼 수 없습니다.
- **한계 품질 이득 대비 증분 비용으로 승격을 정렬합니다.** 효율은
  `예측 이득 / (증분 비용 / 평균 light 비용)`이며 배치 크기와 무관합니다.
- **내용 기반 묶음 단위로 배분합니다.** 묶음 열쇠는 양자화한 dense 프롬프트
  특징과 효율 구간(octave)뿐입니다. 묶음은 통째로 승격하거나 통째로
  남으므로 `episode_id`, `challenge_id`, `split`, 입력 위치나 순서 의존
  동률 처리가 들어갈 자리가 없습니다.
- **불확실하거나 이득이 없는 문항은 싼 모델에 남깁니다.** 품질 마진
  임계값과 문항별 두 가지 꼬리 상한을 모두 통과해야 후보가 됩니다.
  - `max_step_ratio`는 **같은 프롬프트에서** 상위 모델의 예측 비용이 light
    대비 몇 배인지를 제한합니다. 분모가 배치 통계가 아니라 공개 정책이 정한
    문항별 값이므로, 확장이 큰 프롬프트만 모인 배치에서는 승격 자체가 줄어드는
    쪽으로 안전하게 무너집니다.
  - `max_step_load`는 예측 증분을 **배치 평균 light 비용**의 배수로
    제한합니다. 긴 프롬프트 하나가 등급 예산의 큰 몫을 차지할 수 없고, 배치
    크기에 무관하므로 880문항과 1,760문항에서 같게 동작합니다.
- **`axk1-think`는 Premium에서만, `ax31`에서만 승격합니다.** 그마저도 재량
  예산의 정해진 비율까지만 쓰며, 값싸고 예측이 안정적인 `ax31` 승격을 먼저
  모두 처리한 뒤에 남는 예산으로만 배분합니다. think 단계에도 자체
  `max_step_load`가 걸려 있습니다.

등급별 안전 목표는 예측 비용 기준 Fast `1.14`, Balanced `1.60`, Premium
`3.20`입니다. 비용 추정을 보수적으로 잡았기 때문에 실제 비용 비율은 이보다
낮게 나옵니다. 목표값과 두 상한은 아래 **구성 스트레스** 절의 재표본 근거로
보정했으며, 전체 split 평균만으로 고르지 않았습니다.

Fast의 상한이 가장 빡빡한 이유는 여유 폭이 가장 좁기 때문입니다. 880문항
배치에서 승격된 문항 하나가 자기 light 생성의 약 `55배`를 실제로 써 버리면
Fast 비율이 약 `0.06` 움직이는데, 이는 실제 비율과 `1.25` 한도 사이 거리의
대부분입니다. Balanced와 Premium은 같은 문항을 흡수할 수 있으므로 상한을
느슨하게 두고 품질을 더 가져갑니다.

```console
for tier in fast balanced premium; do
  PYTHONPATH=src python3 baselines/safe_margin.py \
    --input data/materialized/dev/inputs.json \
    --tier "$tier" \
    --output "build/safe-margin/$tier.json"
done
```

`--artifact`를 생략하면 모듈 옆의
[`hash-regex-public.v1.json`](hash-regex-public.v1.json)을 사용합니다. 다른
artifact를 시험할 때만 경로를 지정하십시오.

### 제출 컨테이너

제출 이미지가 실행하는 정책이 바로 이 모듈입니다.
[`../container/entrypoint.py`](../container/entrypoint.py)는 정책을 복제하지
않고 `safe_margin.main`을 그대로 호출하며,
[`../container/Dockerfile`](../container/Dockerfile)은
`safe_margin.py`, `hash_regex.py`와 공개 artifact를 `/opt/router/baselines/`에
함께 넣습니다. 따라서 개발용 실행기·스트레스 도구와 제출 컨테이너의 결정이
같은 구현 하나에서 나옵니다. 운영자 호출 인자는
[`../docs/RUNTIME.md`](../docs/RUNTIME.md)의 `--input`, `--tier`, `--output`
그대로이며 `--artifact`는 필요하지 않습니다. 진입점이 safe-margin으로
연결되는지, 이미지에 들어가는 파일이 정확히 무엇인지, 컨테이너 경로와 직접
호출의 출력이 바이트 단위로 같은지는
[`../tests/test_container_entrypoint.py`](../tests/test_container_entrypoint.py)가
Docker 없이 검사합니다.

공식 실행 경로는 이미지의 `ENTRYPOINT`뿐입니다. `setup.cfg`의 `router-run`
console script는 상위 저장소 기준 그대로 `ossp_router.heuristic:main`을
가리키므로, 로컬에서 패키지를 설치해 `router-run`을 직접 부르면 제출 정책이
아니라 참고용 약한 baseline이 실행됩니다. 제출 이미지에는 이 패키지를 설치하지
않으며 해당 명령도 존재하지 않습니다.

### 개발용 한 번 실행

[`../tools/run_mvp.py`](../tools/run_mvp.py)는 세 등급 제출 생성, 공식
self-check 채점, baseline 비교 출력을 한 명령으로 처리합니다. 이 도구는
개발 전용이며 제출 컨테이너에 넣지 않습니다. 제출 컨테이너는 이 도구를 거치지
않고 `safe_margin.main`을 직접 실행하지만, 두 경로 모두 같은 구현을 부르므로
등급별 결정은 동일합니다.

```console
PYTHONPATH=src python3 tools/run_mvp.py
```

`build/mvp/{fast,balanced,premium}.json`과 `build/mvp/report.json`을 만들고,
등급별 실제 비용 비율·품질·가중 최종 점수·모델 선택 분포를 출력합니다.
`--split train`으로 공개 Train에서도 같은 확인을 할 수 있습니다.

### 구성 스트레스(개발 전용)

[`../tools/stress_safe_margin.py`](../tools/stress_safe_margin.py)는 전체
split 평균이 감추는 질문에 답합니다. **같은 종류의 프롬프트라도 구성 비율이
달라지면 실제 비용 비율이 얼마나 움직이는가.**

절차는 결정적 비모수 부트스트랩입니다. 공개 문항을 한 번만 예측해 두고,
seed에서 만든 난수로 split과 같은 크기의 표본을 복원 추출한 뒤, 재표본마다
`plan_selection`을 다시 돌리고 공개 outcome으로 채점합니다. 라우터에는
프롬프트에서 나온 예측만 들어가므로 `episode_id`, split 이름, 벤치마크,
입력 위치, 채점 결과는 선택에 닿지 않습니다. 분위수는 nearest-rank이며
seed·재표본 수·분위수·최댓값·등급별 한도 초과 건수를 보고서에 남깁니다.

```console
PYTHONPATH=src python3 tools/stress_safe_margin.py --split dev
PYTHONPATH=src python3 tools/stress_safe_margin.py --split train
```

split마다 난수를 새로 seed하므로 어느 split을 먼저 돌렸는지에 결과가
의존하지 않습니다.

## 공개 Dev 비교

현재 공개 Dev 880문항의 검증 결과입니다. 각 등급 칸은 `점수 / 실제 비용 비율`
이며, 비용 한도는 Fast `1.25`, Balanced `2.0`, Premium `4.0`입니다. 다섯
구현 모두 세 등급의 예산을 통과합니다.

| Baseline | Fast | Balanced | Premium | 가중 최종 점수 |
| --- | ---: | ---: | ---: | ---: |
| all-light | 0.619318 / 1.000000 | 0.619318 / 1.000000 | 0.619318 / 1.000000 | 0.619318 |
| prompt-heuristic | 0.625852 / 1.072334 | 0.658239 / 1.367866 | 0.691761 / 2.102044 | 0.655341 |
| feature-budget | 0.621023 / 1.038210 | 0.623580 / 1.334059 | 0.691761 / 2.102044 | 0.643011 |
| hash-regex | 0.663068 / 1.235989 | 0.693750 / 1.961506 | 0.740057 / 3.985205 | 0.695369 |
| safe-margin | 0.644886 / 1.092453 | 0.684375 / 1.471480 | 0.699716 / 2.517187 | 0.673182 |

safe-margin은 hash-regex보다 가중 최종 점수가 `0.022` 낮지만, 한도 대비
사용률이 Fast `87.4%`, Balanced `73.6%`, Premium `62.9%`로 훨씬 낮습니다.
hash-regex는 Premium에서 한도의 `99.6%`를 사용했고 사전 검증에서 실제로
한도를 넘어 그 등급이 `0`점 처리되었습니다. 같은 일이 일어나면 hash-regex의
가중 점수는 `0.473`으로 떨어지지만 safe-margin은 `0.673`을 유지합니다.
이 여유가 safe-margin이 의도적으로 품질을 조금 포기하고 얻은 값입니다.

같은 정책을 공개 Train 1,760문항에 적용하면 Fast `0.637358 / 1.098738`,
Balanced `0.681818 / 1.397907`, Premium `0.711648 / 2.581280`이고 가중 최종
점수는 `0.672983`입니다.

### 구성 스트레스 결과

seed `20260822`, 재표본 `500`회, nearest-rank 분위수, split과 같은 표본 크기,
복원 추출입니다. 값은 실제 비용 비율입니다.

| Split | 등급 | 한도 | 전체 | p50 | p95 | p99 | 최댓값 | 한도 초과 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dev | Fast | 1.25 | 1.0925 | 1.1004 | 1.1383 | 1.1478 | 1.1646 | 0 / 500 |
| Dev | Balanced | 2.0 | 1.4715 | 1.4701 | 1.6307 | 1.7040 | 1.7701 | 0 / 500 |
| Dev | Premium | 4.0 | 2.5172 | 2.5425 | 2.9202 | 3.0515 | 3.2110 | 0 / 500 |
| Train | Fast | 1.25 | 1.0987 | 1.0976 | 1.1056 | 1.1089 | 1.1110 | 0 / 500 |
| Train | Balanced | 2.0 | 1.3979 | 1.3954 | 1.4842 | 1.5182 | 1.5761 | 0 / 500 |
| Train | Premium | 4.0 | 2.5813 | 2.5864 | 2.8114 | 2.9050 | 2.9941 | 0 / 500 |

보정 전 정책은 같은 절차에서 Dev Fast가 `500`회 중 `76`회, Dev Premium이
`1`회 한도를 넘었고 Dev Fast 최댓값이 `1.4371`이었습니다. 원인은 예측
정확도 전반이 아니라 **한 문항의 쏠림**이었습니다. 보정 전 Dev Fast에서
승격된 문항 하나가 재량 비용의 `41.3%`를 혼자 썼고, 재표본이 그 문항을 두 번
이상 뽑은 `146`회에서만 초과가 발생했습니다. 한 번 이하로 뽑은 `354`회에서는
초과가 한 건도 없었습니다.

그 문항은 프롬프트만으로는 보이지 않습니다. 예측 증분은 배치 평균의 `0.81`배로
중앙값보다 작은데 실제 증분은 `55.8`배였습니다. 예측 이득과 실제 비용 폭증의
순위 상관은 Dev `-0.07`, Train `-0.06`으로 사실상 없습니다. 즉 **어떤 문항이
터질지는 프롬프트만 보고 고를 수 없습니다.** 보정은 폭증할 문항을 찾아내는
대신, 폭증했을 때 한 문항이 가져갈 수 있는 몫을 두 상한으로 줄이고 Fast의
승격 수준을 낮추는 쪽을 택했습니다.

### 이 근거의 한계

위 표는 **공개 split의 프롬프트 구성을 재표본한 근거이며, 비공개 평가셋의
예산 통과를 보장하지 않습니다.** 부트스트랩은 공개 자료에 이미 들어 있는
문항만 다시 뽑으므로, 공개 split에 없는 프롬프트·벤치마크·토큰 분포는
설명하지 못합니다. 특히 다음은 여전히 열려 있는 위험입니다.

- 공개 자료의 폭증 문항보다 **예측 확장 배수가 낮으면서 실제로는 더 크게
  터지는** 문항이 비공개 자료에 있으면 `max_step_ratio`를 그대로 통과합니다.
- Fast는 여전히 여유 폭이 가장 좁은 등급이며, 이 정책의 주된 잔여 위험입니다.
- 표의 `0 / 500`은 이 seed와 이 절차에서 관측된 값이지 확률적 보증이 아닙니다.

hash-regex의 전체 보고서는
[`hash-regex-public-dev-report.v1.json`](hash-regex-public-dev-report.v1.json)에
있습니다. 안전계수를 이 Dev 자료로 보정했으므로 이 비교는 공개 자료에서의
동작 확인값이며 비공개 최종 평가 성능 추정치가 아닙니다.

공개 baseline은 구현과 학습 방법을 보여주는 예시이며, 채점용 평가셋에서의
예산 통과를 보증하지 않습니다. 공개 Dev에서 Premium 비용 비율이 `3.985`였던
hash-regex baseline은 채점용 평가셋을 사용한 사전 검증에서 비용 비율이
약 `4.2`로 나타나 `4.0` 한도를 초과했고, 규칙에 따라 Premium 등급 점수가 `0`으로
계산되었습니다. 입력 특성과 모델별 토큰 사용량에 따라 비용 비율이 달라질 수
있으므로, 한도에 근접한 정책에는 충분한 비용 여유를 두어야 합니다. 비용은
반올림 전 값으로 비교하며, 한도를 조금이라도 초과하면 해당 등급 점수는
`0`입니다.

학습기는 ridge alpha를 out-of-fold 평균 오차로 고르고, 등급별 안전계수 후보는
공식 Decimal scorer로 평가합니다. Train self-check는 학습 적합도 확인값일
뿐 일반화 점수가 아닙니다. 공개 Dev는 회귀계수 학습에 합치지 않고 안전계수와
예산 통과 여부를 정하는 데만 사용합니다. 학습 파일에는 전역 계수, 공개 파일
해시와 집계값만 남습니다.

## 위험 보정 v2 후보(채택되지 않음, 부정적 결과 기록)

[`risk_calibrated.py`](risk_calibrated.py)는 safe-margin을 대체하기 위해
만든 다음 반복 후보입니다. **측정 결과 챔피언 게이트를 통과하지 못했으므로
제출 컨테이너는 계속 safe-margin을 실행합니다.** 이 절은 그 부정적 결과와
근거 산출물을 기록합니다.

### 설계

- **모든 내부 검증 분할은 출처·과제군 그룹 단위입니다.** 학습기는 안전
  게이트와 동일한 결정적 재구성(`tools/risk_validation.py`의
  `reconstruct_families`: AIME·DeepMind Mathematics는 고정된 공개 선택
  파일, 나머지 7개 과제군은 공개 materialized 프롬프트의 결정적 내용
  규칙)으로 9개 과제군 라벨을 만들고, 과제군 전체를 fold에 통째로
  배정합니다(행 수가 큰 과제군부터 현재 가장 가벼운 fold로, 동률은 fold
  번호 순). 어떤 과제군도 한 분할의 학습·검증 양쪽에 나타날 수 없고, 모든
  행은 정확히 한 번 OOF 예측을 받으며, fold 수가 2 미만이거나 과제군
  수보다 많으면 학습이 거부됩니다(fail-closed). 라벨은 개발 전용 근거이며
  제출 런타임 결정 경로에는 닿지 않습니다. 그룹→fold 배정표와 fold별 행
  수는 학습 보고서의 `group_cv` 절에 있습니다.
- 모델별 score head의 차분 대신 두 승격 단계
  (`ax31-light -> ax31`, `ax31 -> axk1-think`)의 **증분 이득을 직접
  회귀**하고, 양의 이득 확률을 별도 head + 그룹 OOF 확률 보정으로
  예측합니다. Platt 대 isotonic 비교는 과제군 2분할(그룹 분리) 교차
  적합의 Brier로 판정하며, 최종 보정은 방법이 정해진 뒤에야 전체 행으로
  다시 적합합니다. 측정 결과 `ax31` 단계는 Platt, `axk1-think` 단계는
  isotonic이 선택되었습니다.
- 모델별 log-cost head에 **방향 대칭 split-conformal 상한 계수**를 더해
  평균이 아닌 상한 추정으로 문항별 guard와 예산 배분을 수행합니다. 같은
  과제군 2분할의 두 방향 각각에서 상대편을 held-out으로 두고 선언
  분위수를 계산한 뒤 **더 보수적인 계수**를 채택하고, 측정 커버리지는
  채택된 계수를 만들지 않은 쪽 과제군에서 잽니다. 선언 커버리지 `0.85`
  대비 측정 커버리지는 `ax31-light 0.860`, `ax31 0.914`,
  `axk1-think 0.852`이며, 단방향 진단도 학습 보고서에 그대로
  남습니다(예: side-A 과제군만으로 보정한 `ax31` 계수는 side-B에서
  커버리지 `0.690`으로, 과제군 이동 아래에서 단방향 conformal이
  깨진다는 증거). 커버리지 미달 artifact는 검증 단계에서 거부되어
  safe-margin으로 대체 실행됩니다.
- 특징·내용 서명·상한 guard(`max_step_ratio`, `max_step_load`)·think
  하위 예산 구조는 safe-margin과 동일하게 유지합니다. 후보 비교(A: 선형
  hash head, B: dense 구조 특징 boosted tree, C: 혼합)는 같은 그룹 fold
  배정을 공유하는 OOF 선택 집합 이득 기준으로 A를 선택했습니다(전체
  지표는 학습 보고서 참고).
- 학습·튜닝은 공개 Train만 사용하고, 공개 Dev는 마지막 채점과 고정
  게이트 측정에만 사용했습니다.

```console
PYTHONPATH=src python3 baselines/train_risk_calibrated.py \
  --input data/materialized/train/inputs.json \
  --outcomes data/train/outcomes.json \
  --artifact build/risk-calibrated/artifact.json \
  --report build/risk-calibrated/train-report.json
```

동결된 학습 산출물은 [`risk-calibrated-public.v2.json`](risk-calibrated-public.v2.json),
학습 보고서는 [`risk-calibrated-train-report.v2.json`](risk-calibrated-train-report.v2.json)에
있습니다.

### 1차 산출물의 방법 결함과 2차 보정

이 후보의 1차 산출물(커밋 `4c359ab`)은 학습기의 모든 내부 검증 분할을 행
단위로 나눴습니다: OOF fold는 `행 번호 % fold 수`, Platt/isotonic 비교와
conformal 보정·커버리지 측정은 짝/홀 행이었습니다. 공개 Train은 같은
출처·과제군의 유사 문항을 여럿 포함하므로 이는 요구된 출처·과제군 그룹
CV를 위반하고, 같은 과제군이 학습·검증 양쪽에 들어가 모든 검증 수치를
낙관적으로 부풀렸습니다. 2차 보정으로 위 설계 절의 그룹 CV·그룹 분리
보정·방향 대칭 conformal로 재학습하자 다음이 드러났습니다.

- 이득 예측의 OOF 상관은 행 단위 CV의 약 `0.08~0.10`에서 그룹 CV의
  `-0.06 ~ -0.09`로 떨어졌습니다. 즉 **현재 특징으로는 처음 보는
  과제군에 대한 이득 예측력이 사실상 없으며**, 1차 수치는 과제군 내부
  유사 문항 암기가 만든 착시였습니다. 세 head의 ridge 강도는 모두 최대
  후보(3000)로 이동했습니다.
- 짝/홀 conformal 계수 `1.60~2.21`은 그룹 대칭 계수 `2.46~2.84`로
  보수화되었습니다. 단방향 진단이 보여 주듯(`ax31` side-A→side-B
  커버리지 `0.690`) 행 단위 측정 커버리지는 과제군 이동을 전혀 대표하지
  못했습니다.
- 보수화된 상한 아래에서 planner는 고정 spend 목표에 도달하지 못하고
  목표 탐색이 천장에 고정됩니다(Train 실측 Fast `1.036` 대 목표
  `1.099`, Premium `1.754` 대 목표 `2.581`). 지출이 줄어든 만큼 품질도
  내려가 Dev 가중 점수는 1차 `0.679063`에서 `0.658182`가 되었습니다.

기록 보존을 위해, 1차(누수 CV) artifact가 받았던 5,000회 재표본 안전
게이트의 실패는 총 **16건**이었습니다(당시 README는 이 중 2건만
서술했는데, 이는 축소였습니다). Dev 13건: Fast bootstrap 한도 초과
`614/5000`, Fast p99 `1.3632`(기준 `1.1474`), Fast max `1.5009`(기준
`1.1673`), Fast 과제군 holdout **9건 전부**(aime `1.1737`, babilong
`1.0953`, belebele-ko `1.1582`, cruxeval `1.1459`, deepmind-mathematics
`1.1118`, gsm8k `1.1854`, hrmcr `1.1547`, ruletaker `1.1850`, truthfulqa
`1.1592` — 모두 기준 대비 허용 오차 `0.005` 초과), Balanced holdout
ruletaker `1.4074`(기준 `1.3945`). Train 3건: Fast holdout cruxeval
`1.0967`(기준 `1.0838`), deepmind-mathematics `1.0920`(기준 `1.0834`),
truthfulqa `1.0994`(기준 `1.0858`). 이 가운데 Dev Fast의 12건 — 재표본
3건과 deepmind-mathematics를 제외한 holdout 8건 — 은 **같은 폭증 문항
하나**(dev-0678, deepmind-mathematics 과제군, 실제 증분이 Fast 재량
지출의 `41.4%`)를 선택에 포함한 측정이라는 공통 원인을 공유합니다.
폭증 문항이 빠지는 deepmind-mathematics holdout(`1.1118` 대 기준
`1.1010`)과 Dev Balanced ruletaker, Train Fast holdout 3건은 폭증
문항과 무관하게 1차 후보의 전반적 지출 성향이 허용 오차를 넘은
잔여 실패였습니다.

### 개발 전용 진단·검증 도구

- [`../tools/oracle_headroom.py`](../tools/oracle_headroom.py)는 공개
  outcome을 사용하는 **개발 전용** oracle 분해로, 어떤 예측(품질/비용)이
  병목인지와 안전 범위 안에서 도달 가능한 상한을 측정합니다. 제출 런타임
  경로에서는 절대 실행되지 않으며, 격리는
  [`../tests/test_oracle_headroom.py`](../tests/test_oracle_headroom.py)가
  검사합니다.
- [`../tools/risk_validation.py`](../tools/risk_validation.py)는 동일 재표본
  아래에서 기준(safe-margin)과 후보를 비교하는 고정 안전 게이트입니다.
  seed `20260825`, 5,000회 재표본, nearest-rank 분위수, 그리고 공개
  materialization 입력에서만 재구성한 출처·과제군 holdout을 사용합니다.
  게이트: 모든 등급 한도 초과 `0`, 후보 p99·최댓값이 같은 실행의 기준보다
  `0.005`(문서화된 허용 오차) 이상 나쁘지 않을 것, 모든 과제군 holdout이
  한도를 지키고 기준 대비 같은 오차 안일 것.

```console
PYTHONPATH=src python3 tools/oracle_headroom.py --split dev
PYTHONPATH=src python3 tools/risk_validation.py --split train --split dev \
  --candidate-artifact baselines/risk-calibrated-public.v2.json
```

### 측정 결과: oracle 분해 (seed 20260825, 5,000회 재표본)

공개 Dev 가중 점수 기준입니다. 전체 보고서는
[`oracle-headroom-dev.v1.json`](oracle-headroom-dev.v1.json)과
[`oracle-headroom-train.v1.json`](oracle-headroom-train.v1.json)에 있습니다.

| 변형 | Dev 가중 점수 |
| --- | ---: |
| 현재 예측 + 현재 라우터 (기준) | 0.673182 |
| oracle 품질 이득 + 현재 비용 예측 | 0.719517 |
| 현재 품질 예측 + oracle 실제 비용 | 0.674687 |
| oracle 품질 + oracle 비용 | 0.718750 |
| 안전 범위 oracle 상한 (달성 가능/LP 상한) | 0.772898 / 0.773781 |

정지 게이트(`LP 상한 >= 0.690000`)는 통과했으므로 학습 후보를 진행했습니다.
분해가 보여 주는 병목은 **품질 이득 예측**입니다: 비용만 oracle로 바꾸면
`+0.0015`뿐이지만 품질 이득만 oracle로 바꾸면 지출을 줄이면서도 `+0.0463`이
됩니다. 그런데 그룹 CV로 정직하게 잰 실제 학습 가능한 이득 예측의 OOF
상관은 `-0.06 ~ -0.09`로(학습 보고서의 후보 비교 절), 이 병목은 현재
특징·자료 규모에서는 넘을 수 없는 것으로 측정되었습니다.

### 측정 결과: v2 후보 대 safe-margin

각 칸은 `점수 / 실제 비용 비율`입니다.

| Split | 라우터 | Fast | Balanced | Premium | 가중 점수 |
| --- | --- | ---: | ---: | ---: | ---: |
| Dev | safe-margin | 0.644886 / 1.092453 | 0.684375 / 1.471480 | 0.699716 / 2.517187 | 0.673182 |
| Dev | risk-calibrated v2 | 0.631250 / 1.037057 | 0.663636 / 1.257724 | 0.688636 / 1.675448 | 0.658182 |
| Train | safe-margin | 0.637358 / 1.098738 | 0.681818 / 1.397907 | 0.711648 / 2.581280 | 0.672983 |
| Train | risk-calibrated v2 | 0.628267 / 1.036313 | 0.664062 / 1.189805 | 0.682386 / 1.753677 | 0.655241 |

### 측정 결과: 5,000회 재표본 안전 게이트 — 통과 (78/78)

전체 보고서는 [`risk-validation-report.v2.json`](risk-validation-report.v2.json)에
있습니다(seed `20260825`, 5,000회 재표본, 허용 오차 `0.005`). 2차 보정
후보는 split당 39개 검사(등급별 bootstrap 초과·p99·max·전체 예산, 그리고
9개 과제군 holdout) **전부를 통과**했습니다. 요약(실제 비용 비율):

| Split | 등급 | 기준 p99 | 후보 p99 | 기준 max | 후보 max | 후보 한도 초과 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Dev | Fast | 1.1474 | 1.0477 | 1.1673 | 1.0567 | 0 / 5000 |
| Dev | Balanced | 1.6919 | 1.4636 | 1.8706 | 1.6272 | 0 / 5000 |
| Dev | Premium | 3.0118 | 1.9408 | 3.4176 | 2.1373 | 0 / 5000 |
| Train | Fast | 1.1090 | 1.0425 | 1.1151 | 1.0455 | 0 / 5000 |
| Train | Balanced | 1.5237 | 1.2220 | 1.6214 | 1.2473 | 0 / 5000 |
| Train | Premium | 2.8966 | 1.9877 | 3.1491 | 2.1623 | 0 / 5000 |

통과의 이유는 예측력이 아니라 지출 축소입니다: 보수화된 conformal 상한
아래에서 후보는 모든 등급에서 기준보다 훨씬 적게 쓰므로(예: Dev Fast 전체
`1.037` 대 기준 `1.092`) 꼬리와 holdout이 전부 기준 범위 안에 들어가고,
1차 실패를 만든 폭증 문항 dev-0678도 더 이상 Fast에서 승격되지 않습니다
(Fast 재량 지출 집중도 `0.414 -> 0.057`). 다만 Dev Balanced에서는 한
문항이 재량 지출의 `24.6%`를 차지하는 집중이 남아 있으며, 게이트는
집중도를 직접 제한하지 않으므로 이는 잔여 위험으로 기록합니다.

### 결론(부정적 결과)

- 챔피언 게이트(가중 Dev `>= 0.690000` + 전체 안전 게이트 통과)는 달성하지
  못했습니다: 안전 게이트는 78건 전부 통과했지만 측정된 Dev 가중 점수가
  `0.658182`로 목표 `0.690000`은 물론 safe-margin의 `0.673182`에도
  미달합니다.
- 그룹 CV로 정직하게 재면 현재 특징의 과제군 밖 이득 예측 상관은
  `-0.06 ~ -0.09`, 즉 예측력이 없습니다. 1차 산출물의 우위(가중
  `0.679063`)는 행 단위 CV 누수가 만든 것이었고, 누수를 제거하면 같은
  위험 예산 안에서 safe-margin을 이길 근거가 사라집니다. oracle 분해가
  가리키는 품질 이득 예측 병목은 현재 자료 규모에서 넘을 수 없는 것으로
  측정되었습니다.
- 따라서 제출 기본값은 safe-margin을 유지하고, 이 후보는 검증 도구·진단
  보고서와 함께 비채택 기록으로 남깁니다. `risk_calibrated.py`를 직접
  실행하면 artifact 검증 실패 시 safe-margin으로 결정적으로 대체 실행되며,
  이는 [`../tests/test_risk_calibrated_router.py`](../tests/test_risk_calibrated_router.py)가
  검사합니다.

## 표현 감사 v1: 과제군 밖 증분 이득

후속 실험은 [`../tools/representation_audit.py`](../tools/representation_audit.py)와
[`representation_features.py`](representation_features.py)에 고정했습니다. 공개
Train의 재구성 가능한 9개 출처·과제군을 하나씩 통째로 제외하는 LOFO 예측만으로
표현과 ridge 강도(`100`, `1000`, `3000`)를 선택합니다. 런타임 특징에는 prompt
또는 messages의 role/content만 들어가며, source/family, episode ID, 행 위치,
split, outcome과 Dev outcome은 들어가지 않습니다. 증분 비용도 같은 LOFO에서
모델별 log-cost로 예측하고, 고정 safe-margin Train 지출에서 실제 선택 이득과
실제 증분 비용을 측정합니다.

비교 표현은 다음과 같습니다.

- A: 기존 14개 dense + 256개 signed word unigram/bigram hash
- B: 36개 확장 구조 특징. field/role별 길이·비율, role 전환, 문단/선택지/코드·수식
  모양, 그리고 protocol 입력에서 유도 가능한 context/question 경계를 포함
- C: 기존 dense 14개 + field/role-aware word 1/2-gram 및 character 3/4/5-gram
  signed hash 256개
- D: B와 semantic proxy hash의 결합(292개)

감사 기준인 A는 기존 구현을 정확히 유지하므로 입력 길이를 제한하지 않습니다.
후보 B/C/D는 dense와 structural 구성 요소를 포함해 모든 prompt/message field를
각각 최대 32,768문자로 제한하며, semantic proxy는 FNV-1a signed hashing만
사용합니다. 네트워크, 외부 API, 다운로드한 weight와 런타임 패키지가 없으며,
전체 공개 split 추출 완료, 특징 1,024개, 5-head artifact 추정 1,000,000 byte를
fail-closed runtime gate로 사용합니다. 채택될 경우에만 공식 90초 격리 benchmark를
추가로 통과해야 합니다.

### 사전 선언 채택 게이트

새 표현은 두 승격(`ax31-light -> ax31`, `ax31 -> axk1-think`) 각각에서 Train
LOFO 상관 `>= 0.02`, 양의 family 상관 `>= 6/9`, family별 선택 이득 최솟값
`>= -0.005`를 모두 만족해야 합니다. 또한 Fast/Balanced/Premium의 모든 해당
고정 지출점에서 기존 A보다 문항당 실제 선택 이득이 `>= 0.002` 높고 runtime
gate를 통과해야 합니다. 그 다음에만 기존 그룹 분리 calibration/conformal,
5,000회 안전 검사, `0.005` 허용 오차와 Dev 가중 점수 `>= 0.690000` champion
gate를 엽니다. 어느 단계든 불확정 또는 실패면 제출 기본값은 safe-margin입니다.

### Train LOFO 결과와 결정

각 이득은 전체 Train 문항 수로 나눈 실제 선택 증분 점수이며, 비용은 all-light
실제 비용 대비 선택 집합의 signed 증분 비용입니다.

| 표현 | light→ax31 상관 | 양의 family | Fast 이득 / 비용 | Balanced 이득 / 비용 | ax31→think 상관 | 양의 family | Premium 이득 / 비용 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A 기존 | -0.060370 | 2/9 | 0.007102 / 0.040548 | 0.030966 / 0.339717 | 0.036664 | 4/9 | 0.003977 / 0.414624 |
| B 확장 구조 | 0.033517 | 5/9 | 0.018324 / 0.040373 | 0.045028 / 0.339544 | 0.135148 | 4/9 | 0.001705 / 0.426026 |
| C semantic proxy | -0.040637 | 5/9 | 0.012358 / 0.040388 | 0.041051 / 0.339556 | 0.021195 | 3/9 | 0.004972 / 0.414881 |
| D 구조+semantic | -0.026074 | 5/9 | 0.014347 / 0.040374 | 0.038636 / 0.339532 | 0.151465 | 5/9 | 0.004261 / 0.415200 |

Train 합산 선택 이득으로 동결한 진단 선두는 B입니다. 그러나 두 승격 모두
양의 family가 각각 `5/9`, `4/9`뿐이고, Premium 이득 `0.001705`는 기존 표현에
요구한 `0.005977`보다 낮아 채택 gate가 실패했습니다. family별 선택 이득
최솟값 `-0.003125`는 바닥값을 통과했습니다. 따라서 calibration/conformal
재학습, runtime artifact 통합과
새 후보의 안전/champion gate는 열지 않았습니다.

동결 뒤 단 한 번 수행한 공개 Dev 진단에서 B는 light→ax31 상관 `0.084931`,
Fast `0.027273 / 0.051936`, Balanced `0.050000 / 0.351032`; ax31→think 상관
`0.316753`, Premium `0.017045 / 0.478395`였습니다. 이 사후 수치는 Train 선택을
바꾸지 않으며, 전체 라우터 가중 점수도 아니고 private-set 일반화 근거도
아닙니다. 재현 가능한 전체 family별 수치, hash와 gate 실패는
[`representation-audit-report.v1.json`](representation-audit-report.v1.json)에
있습니다. 기존 v2의 안전 결과(78/78 통과)와 Dev `0.658182`, safe-margin Dev
`0.673182`는 그대로이며 제출 기본값도 safe-margin입니다.

## Semantic upgrade-event experiment v1

[`../configs/semantic-upgrade-events-protocol.v1.json`](../configs/semantic-upgrade-events-protocol.v1.json)은
새 후보의 Dev outcome을 읽기 전에 genuine multilingual encoder registry, artifact
hash, 세 갈래 win/tie/loss target, step별 nested family CV, OOD abstention, matched
spend, 채택 기준과 fail-closed 동작을 고정합니다. 기존 `C-semantic-proxy`는
word/character n-gram hash이며 genuine pretrained embedding baseline이 아닙니다.

고정 registry의 유일한 후보는 MIT 라이선스
`intfloat/multilingual-e5-small@5697a65b0a002a92fe8c4fc9d495303ffff9c7d2`입니다.
필요한 architecture-neutral ONNX와 tokenizer 6개 파일은 총 492,421,554 bytes이고
모두 immutable URL, 크기, SHA-256으로 고정했습니다. 공개된 더 작은 quantized
ONNX는 AVX512 VNNI 전용이라 공식 `linux/arm64` 후보에서 제외했습니다.

고정 revision의 registry 파일 6개는 Git이 무시하는 로컬 경로
`.local-data/semantic-encoder/`에만 내려받아 크기와 SHA-256을 모두 확인합니다.
weight는 commit하지 않고 제출 이미지에도 넣지 않습니다. Python 3.11.15 격리
환경에는
[`semantic-upgrade-events-requirements.txt`](../configs/semantic-upgrade-events-requirements.txt)의
`numpy 2.0.2`, `onnxruntime 1.22.1`, `tokenizers 0.21.4`, `psutil 7.0.0`만 extraction
요구 사항으로 고정했습니다. 네트워크는
artifact provisioning과 이미지 빌드 단계에서만 쓰고 evaluation runtime에는
필요하지 않습니다.

설치 파일 합계는 wheel이 플랫폼별로 다르므로 환경마다 달라집니다. 근거 파일에
기록된 값은 공식 `linux/arm64` 측정 환경이 `98,073,668 bytes`
(`dependencies.required_installed_bytes`), 별도로 기록한 native Apple arm64
preflight 환경이 `176,724,066 bytes`
(`native_apple_arm64_preflight.dependencies.required_installed_bytes`)입니다. 두
값은 서로 다른 환경의 관측이므로 합치거나 대체해 쓰지 않습니다. 위 4개 직접
요구 사항은 version-pinned이지만, 실제로 해결된 transitive 환경 전체는 근거
파일의 `dependencies.resolved_environment`에 이름과 version으로 기록만 되어
있을 뿐 hash-lock되어 있지는 않습니다. 따라서 향후 재빌드에서 transitive
의존성까지 동일하게 재현된다고 주장하지 않습니다.

모든 prompt/message content field는 tokenization 전에 각각 32,768자로 자르고,
모델 카드가 feature embedding에 요구한 `query: ` prefix를 붙입니다. 512 token
상한, attention-mask mean pooling, L2 normalization, ONNX sequential execution,
intra-op 2/inter-op 1 thread를 사용합니다. 길이가 크게 다른 prompt의 padding
비용을 피하려고 batch size 1을 동결했습니다.

### 측정 환경 세 가지를 구분합니다

| 환경 | 무엇을 증명하나 | 공식 feasibility gate |
| --- | --- | --- |
| native Apple arm64 preflight (`darwin/arm64`) | 추출이 동작하고 Train-only 실험을 열어도 되는지 | protocol상 절대 통과시키지 않음 |
| 로컬 Colima `linux/arm64` 컨테이너 | 공식 **아키텍처**에서 동결 한도 안에 들어가는지 | 통과 가능 — 이번에 측정 |
| 운영자 최종 대회 장비 | 최종 자원 여유와 동점 레이턴시 | 운영자만 측정 |

로컬 Colima VM은 공식 아키텍처(`linux/arm64`)와 같고 커널이 같은 cgroup v2
한도를 실제로 강제하지만, QEMU 기반 가상 머신이며 운영자의 최종 대회 장비가
아닙니다. 이 저장소는 어디에서도 로컬 Colima VM을 최종 대회 장비라고 주장하지
않습니다. 운영자 측 장비와 절차는
[`../docs/APPLE_SILICON_MEASUREMENT.md`](../docs/APPLE_SILICON_MEASUREMENT.md)에
있습니다.

### 공식 아키텍처 `linux/arm64` 측정 결과

전용 실험 이미지
[`../container/semantic-measurement.Dockerfile`](../container/semantic-measurement.Dockerfile)로
`linux/arm64` 이미지를 만듭니다. base는
`python:3.11.15-slim-bookworm@sha256:d29f48a31a8b408ed19272ca1e7b10ebae13b240a27e862d3d4217c528e2e0c3`,
빌드한 image id는
`sha256:7615fd87c6c99af1344b52d000ec6fbe5965c5e11dbf3c5ed0bcd0a68ddf976a`입니다.
이미지는 동결 artifact 6개와 pinned extraction 의존성만 담고, 빌드 마지막
단계에서 6개 파일의 크기와 SHA-256을 다시 확인합니다. 이 이미지는 제출
이미지([`../container/Dockerfile`](../container/Dockerfile))와 baseline 런타임
벤치마크 이미지([`../container/measurement.Dockerfile`](../container/measurement.Dockerfile))
어느 쪽도 바꾸지 않으며 제출 경로에 들어가지 않습니다.

측정 실행은 CPU 2개, 메모리 2 GiB, 추가 스왑 없음, 프로세스·스레드 32개,
네트워크 없음, 읽기 전용 루트입니다. 컨테이너 안에서 커널이 알려 준 값
(cgroup v2, kernel `6.8.0-117-generic`, Docker Engine 29.5.2, Colima 0.10.3,
Apple M2 / macOS 26.3 호스트):

| 관측 | 값 | 동결 한도 |
| --- | ---: | ---: |
| 1,760행 1차 추출 | 83.352049초 | 90초 |
| 1,760행 2차 추출 | 84.702307초 | 90초 |
| cgroup `memory.peak` | 1,221,636,096 bytes | 2,147,483,648 bytes |
| 프로세스 peak RSS | 1,252,179,968 bytes | 2,147,483,648 bytes |
| cgroup `pids.peak` / 최대 thread+PID 관측 | 7 / 7 | 32 |
| cgroup `cpu.max` | `200000 100000` (2 core) | 2 core |
| cgroup `memory.swap.max` | `0` | 추가 스왑 없음 |
| network interface | `lo` 하나 | 없음 |

두 pass의 출력은 byte-identical이고 두 SHA-256이 모두
`ad094b73df60cba480d148d5246a491f1925355f8bf29c40510b019ceae74726`, 최대 norm
오차는 `1.192093e-7`이었습니다. 따라서 protocol이 요구한 공식 아키텍처
feasibility gate는 이번에 처음으로 **측정되어 통과**했습니다. 이는 아키텍처
측정이며 운영자 최종 장비의 여유나 동점 레이턴시를 대신하지 않습니다.

같은 두 pass를 native Apple arm64(M2, macOS 26.3, 컨테이너 없음)에서도
preflight로 돌려 76.122452초와 75.757317초, byte-identical을 확인했습니다. 이
preflight는 cgroup 한도를 강제하지 않으므로 protocol 정의상 공식 gate를 열지
않으며, 근거 파일에도 `qualifies_official_feasibility=false`로 남습니다.

### Train-only 결과

Train-only nested family-disjoint CV는 위 `linux/arm64` embedding 행렬로
완주했습니다. fold별로 class imbalance weight, scaling, 64차원
training-variance projection, win/tie/loss event head, conditional win/loss
magnitude, log-cost head, OOD threshold와 hyperparameter를 fit했습니다.

| 승격 | OOF 상관 | 양의 family | Fast 이득 / 비용 | Balanced 이득 / 비용 | Premium 이득 / 비용 |
| --- | ---: | ---: | ---: | ---: | ---: |
| light→ax31 | -0.011546 | 3/9 | 0.011506 / 0.098058 | 0.021449 / 0.374902 | — |
| ax31→think | 0.071249 | 5/9 | — | — | -0.000284 / 0.870160 |

첫 승격은 상관, 양의 family, Balanced matched-spend 기준을 실패했고, 둘째는 양의
family와 Premium matched-spend 기준을 실패했습니다. family gain floor는 각각
`0.000000`, `-0.004292`로 통과했습니다. 따라서 이는 setup이나 infrastructure
실패가 아니라 완료된 Train quality-gate 실패입니다. 인프라가 아니라 품질이
문제라는 점은 이번 측정으로 더 분명해졌습니다. 공식 아키텍처 feasibility는
통과했는데도 동결한 Train 채택 기준 9개 중 5개가 실패했고, native 측정과
`linux/arm64` 측정의 matched-spend 값이 같은 자리까지 일치했습니다(OOF 상관만
float kernel 차이로 `1e-8` 자리에서 달라집니다).

그러므로 Dev outcome은 읽지 않았고 calibration/conformal 후속 경로,
5,000-resample safety, 채택 후보를 대상으로 하는 제출 이미지 benchmark gate는
열지 않았습니다. `candidate_adopted=false`, 기본 제출은 safe-margin
그대로입니다. 전체 outer selection, inner objective, OOD coverage와 관측값은
[`semantic-upgrade-events-evidence.v1.json`](semantic-upgrade-events-evidence.v1.json),
결정 보고서는 [`semantic-upgrade-events-report.v1.json`](semantic-upgrade-events-report.v1.json)에
있습니다.

### 재현 절차

보고서만 다시 만들 때는 인자가 필요 없습니다.

```console
PYTHONPATH=src:baselines:tools python3.11 tools/semantic_upgrade_experiment.py
```

측정부터 다시 할 때는 네 단계입니다. 네트워크는 1단계와 2단계에만 쓰고,
3단계 측정 실행에는 쓰지 않습니다.

```console
# 1) 동결 artifact를 무시 경로에 받아 크기와 SHA-256을 확인합니다.
PYTHONPATH=src:baselines:tools python3.11 tools/semantic_upgrade_experiment.py \
  --provision-only --encoder-dir .local-data/semantic-encoder

# 2) 실험 전용 linux/arm64 측정 이미지를 만듭니다.
#    Docker 29의 buildx는 로컬 load 시 --provenance=false가 필요합니다.
docker buildx build --platform linux/arm64 --provenance=false --load \
  --file container/semantic-measurement.Dockerfile \
  --tag ossp-semantic-measurement:v1 .

# 3) 동결 한도 안에서 1,760행 추출을 두 번 수행합니다.
docker volume create ossp-semantic-out
docker run --rm --user 0:0 --entrypoint chown -v ossp-semantic-out:/out \
  ossp-semantic-measurement:v1 -R 65532:65532 /out
docker run --rm --platform linux/arm64 \
  --cpus 2 --memory 2g --memory-swap 2g --pids-limit 32 \
  --network none --read-only --tmpfs /tmp:rw,noexec,nosuid,nodev,size=64m \
  --security-opt no-new-privileges --user 65532:65532 \
  -v ossp-semantic-out:/out \
  -e OSSP_SEMANTIC_ENVIRONMENT_LABEL=local-colima-qemu-linux-arm64 \
  -e OSSP_SEMANTIC_IMAGE_REFERENCE=ossp-semantic-measurement:v1 \
  ossp-semantic-measurement:v1 \
  --extract --encoder-dir /opt/encoder --measurement-dir /out

# 4) 회수한 embedding 행렬에서 근거와 보고서를 다시 만듭니다.
PYTHONPATH=src:baselines:tools python3.11 tools/semantic_upgrade_experiment.py \
  --build-evidence --measurement-dir "$MEASUREMENT_DIR" \
  --native-preflight "$PREFLIGHT_DIR/extraction-measurement.json"
```

`OSSP_SEMANTIC_ENVIRONMENT_DESCRIPTION`, `OSSP_SEMANTIC_IMAGE_ID`,
`OSSP_SEMANTIC_BASE_IMAGE_DIGEST`, `OSSP_SEMANTIC_DOCKERFILE_SHA256`을 함께
넘기면 그 값이 그대로 근거 파일의 환경 증거가 됩니다. native preflight는 같은
도구를 컨테이너 없이 `--extract --extraction-only`로 실행해 만듭니다. 4단계는
컨테이너가 남긴 `train-embeddings.npy`의 SHA-256이 측정 기록과 다르면 거부하며,
같은 입력에서 두 번 실행하면 byte-identical 산출물을 만듭니다.
