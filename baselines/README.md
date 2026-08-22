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
  임계값과 문항별 꼬리 비용 상한(`max_step_ratio`)을 모두 통과해야 후보가
  됩니다.
- **`axk1-think`는 Premium에서만, `ax31`에서만 승격합니다.** 그마저도 재량
  예산의 정해진 비율까지만 쓰며, 값싸고 예측이 안정적인 `ax31` 승격을 먼저
  모두 처리한 뒤에 남는 예산으로만 배분합니다.

등급별 안전 목표는 예측 비용 기준 Fast `1.14`, Balanced `1.70`, Premium
`3.20`입니다. 비용 추정을 보수적으로 잡았기 때문에 실제 비용 비율은 이보다
낮게 나옵니다. 목표값은 공개 Train에서 고르고 공개 Dev로 확인했습니다.

```console
for tier in fast balanced premium; do
  PYTHONPATH=src python3 baselines/safe_margin.py \
    --input data/materialized/dev/inputs.json \
    --artifact baselines/hash-regex-public.v1.json \
    --tier "$tier" \
    --output "build/safe-margin/$tier.json"
done
```

### 개발용 한 번 실행

[`../tools/run_mvp.py`](../tools/run_mvp.py)는 세 등급 제출 생성, 공식
self-check 채점, baseline 비교 출력을 한 명령으로 처리합니다. 이 도구는
개발 전용이며 제출 컨테이너에 넣지 않습니다. 컨테이너 진입 명령
`router-run`은 기존 그대로 유지됩니다.

```console
PYTHONPATH=src python3 tools/run_mvp.py
```

`build/mvp/{fast,balanced,premium}.json`과 `build/mvp/report.json`을 만들고,
등급별 실제 비용 비율·품질·가중 최종 점수·모델 선택 분포를 출력합니다.
`--split train`으로 공개 Train에서도 같은 확인을 할 수 있습니다.

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
| safe-margin | 0.653125 / 1.153642 | 0.684375 / 1.539658 | 0.701136 / 2.600279 | 0.676903 |

safe-margin은 hash-regex보다 가중 최종 점수가 `0.018` 낮지만, 한도 대비
사용률이 Fast `92%`, Balanced `77%`, Premium `65%`로 훨씬 낮습니다.
hash-regex는 Premium에서 한도의 `99.6%`를 사용했고 사전 검증에서 실제로
한도를 넘어 그 등급이 `0`점 처리되었습니다. 같은 일이 일어나면 hash-regex의
가중 점수는 `0.473`으로 떨어지지만 safe-margin은 `0.677`을 유지합니다.
이 여유가 safe-margin이 의도적으로 품질을 조금 포기하고 얻은 값입니다.

같은 정책을 공개 Train 1,760문항에 적용하면 Fast `1.103`, Balanced `1.515`,
Premium `2.591`이고 가중 최종 점수는 `0.681`입니다. 두 분할 사이에서 Fast의
초과분(`비율 - 1`)이 `0.103`에서 `0.154`로 `48.8%` 흔들리는 반면 Balanced는
`4.8%`, Premium은 `0.6%`만 움직입니다. Dev 기준으로 한도까지 남은 초과분
여유는 Fast `62.7%`, Balanced `85.3%`, Premium `87.5%`입니다. 즉 비공개
자료에서 Train 대비 Dev만큼의 비용 상승이 한 번 더 일어나도 세 등급 모두
한도 안에 있습니다. Fast가 가장 빡빡한 등급이며 이 정책의 주된 잔여
위험입니다.

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
