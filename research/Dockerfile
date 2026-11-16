FROM nvcr.io/nvidia/pytorch:25.09-py3

COPY sapai /workspace/sapai
COPY sapai-gym /workspace/sapai-gym
WORKDIR /workspace/sapai-gym
RUN python -m pip install --no-cache-dir -r requirements-dgx.txt \
    && python -m pip install --no-cache-dir --no-deps -e .
ENV PYTHONPATH=/workspace/sapai:/workspace/sapai-gym
ENV PYTHONDONTWRITEBYTECODE=1
