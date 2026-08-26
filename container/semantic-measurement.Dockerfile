# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-FileCopyrightText: Copyright 2026 OSSP 2026 LLM Router Challenge participant
# SPDX-License-Identifier: Apache-2.0
#
# 실험 전용 측정 이미지입니다. 제출 이미지(../container/Dockerfile)와
# baseline 런타임 벤치마크 이미지(../container/measurement.Dockerfile)는
# 이 파일과 무관하며 이 이미지는 제출·평가 경로에 들어가지 않습니다.
# 목적은 동결된 semantic upgrade-event 실험의 추출 실행 가능성을 공식
# 아키텍처 linux/arm64에서 재는 것뿐입니다.
#
# 빌드 전에 고정 revision artifact를 로컬 캐시로 먼저 받습니다(네트워크는
# 이 provisioning 단계와 빌드 단계에만 사용하고 측정 실행에는 쓰지 않습니다).
#   PYTHONPATH=src:baselines:tools python3 tools/semantic_upgrade_experiment.py \
#     --provision-only --encoder-dir .local-data/semantic-encoder
#   docker buildx build --platform linux/arm64 --provenance=false --load \
#     --file container/semantic-measurement.Dockerfile \
#     --tag ossp-semantic-measurement:v1 .

FROM python:3.11.15-slim-bookworm@sha256:d29f48a31a8b408ed19272ca1e7b10ebae13b240a27e862d3d4217c528e2e0c3

LABEL io.sktelecom.ossp.purpose="semantic-upgrade-event experiment extraction measurement" \
      io.sktelecom.ossp.submission-image="false" \
      io.sktelecom.ossp.encoder-revision="5697a65b0a002a92fe8c4fc9d495303ffff9c7d2"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/experiment/src:/opt/experiment/baselines:/opt/experiment/tools \
    TMPDIR=/tmp

WORKDIR /opt/experiment

COPY configs/semantic-upgrade-events-requirements.txt /opt/experiment/configs/
RUN python3 -m pip install --no-cache-dir --no-compile \
        -r /opt/experiment/configs/semantic-upgrade-events-requirements.txt \
    && python3 -m pip freeze > /opt/experiment/pip-freeze.txt

COPY src /opt/experiment/src
COPY tools /opt/experiment/tools
COPY baselines /opt/experiment/baselines
COPY configs /opt/experiment/configs
COPY data/train/outcomes.json /opt/experiment/data/train/
COPY data/train/aime-selection.json /opt/experiment/data/train/
COPY data/materialized/train/inputs.json /opt/experiment/data/materialized/train/
RUN chmod -R a+rX /opt/experiment

# 동결 registry의 6개 파일만 이미지에 넣고, 빌드 시점에 크기와 SHA-256을
# 다시 확인합니다. 실패하면 이미지가 만들어지지 않습니다.
COPY --chmod=0644 .local-data/semantic-encoder/onnx/config.json               /opt/encoder/onnx/
COPY --chmod=0644 .local-data/semantic-encoder/onnx/model.onnx                /opt/encoder/onnx/
COPY --chmod=0644 .local-data/semantic-encoder/onnx/sentencepiece.bpe.model   /opt/encoder/onnx/
COPY --chmod=0644 .local-data/semantic-encoder/onnx/special_tokens_map.json   /opt/encoder/onnx/
COPY --chmod=0644 .local-data/semantic-encoder/onnx/tokenizer.json            /opt/encoder/onnx/
COPY --chmod=0644 .local-data/semantic-encoder/onnx/tokenizer_config.json     /opt/encoder/onnx/
RUN chmod 0755 /opt/encoder /opt/encoder/onnx \
    && python3 /opt/experiment/tools/semantic_upgrade_experiment.py \
        --verify-only --encoder-dir /opt/encoder

USER 65532:65532

ENTRYPOINT ["python3", "/opt/experiment/tools/semantic_upgrade_experiment.py"]
