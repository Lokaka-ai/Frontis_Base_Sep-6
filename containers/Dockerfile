FROM python:3.12.5-slim-bookworm

ARG DEBIAN_FRONTEND=noninteractive
RUN python -m pip install --no-cache-dir \
      numpy==2.1.3 \
      pandas==2.2.3 \
      scikit-learn==1.5.2 \
      scipy==1.14.1 \
    && find /usr/local/lib/python3.12 -type d -name __pycache__ -prune -exec rm -rf '{}' +

RUN mkdir -p /app \
    && useradd --uid 65532 --no-create-home --shell /usr/sbin/nologin candidate
USER 65532:65532
WORKDIR /work
