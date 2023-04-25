FROM swaggerapi/swagger-ui:v4.18.2 AS swagger-ui
FROM nvidia/cuda:11.7.0-base-ubuntu22.04

ENV PYTHON_VERSION=3.10
ENV VENV=/app/.venv

RUN update-ca-certificates --fresh

RUN export DEBIAN_FRONTEND=noninteractive \
    && apt-get -qq update \
    && apt-get -qq upgrade \
    && apt-get -qq install --no-install-recommends \
    build-essential \
    llvm \
    llvm-dev \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-venv \
    python3-pip \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s -f /usr/bin/python${PYTHON_VERSION} /usr/bin/python3 && \
    ln -s -f /usr/bin/python${PYTHON_VERSION} /usr/bin/python && \
    ln -s -f /usr/bin/pip3 /usr/bin/pip

RUN python3 -m venv $VENV \
    && $POETRY_VENV/bin/pip install -U pip setuptools # \
    # && $POETRY_VENV/bin/pip install poetry==1.4.2

ENV PATH="${PATH}:${VENV}/bin"

# RUN pip config set global.trusted-host \
#         "pypi.org files.pythonhosted.org pypi.python.org" \
#         --trusted-host=pypi.python.org \
#         --trusted-host=pypi.org \
#         --trusted-host=files.pythonhosted.org
# RUN pip config set global.http.sslVerify false

WORKDIR /app

# COPY poetry.lock pyproject.toml ./
# COPY pyproject.toml ./
COPY requirements.txt ./

# RUN poetry config virtualenvs.in-project true
# RUN poetry install --no-root
RUN --mount=type=cache,target=/root/.cache/pip \
    $VENV/bin/pip install -r requirements.txt --retries 20 -v --extra-index-url https://download.pytorch.org/whl/cu117

COPY . .
COPY --from=swagger-ui /usr/share/nginx/html/swagger-ui.css swagger-ui-assets/swagger-ui.css
COPY --from=swagger-ui /usr/share/nginx/html/swagger-ui-bundle.js swagger-ui-assets/swagger-ui-bundle.js

# RUN poetry install
# RUN $VENV/bin/pip install torch==1.13.0+cu117 --retries 20 -f https://download.pytorch.org/whl/torch

# RUN pip install --retries 20 .

# CMD $VENV/bin/gunicorn --bind 0.0.0.0:9000 --log-level=debug --workers 1 --timeout 0 app.webservice:app -k uvicorn.workers.UvicornWorker
CMD $VENV/bin/uvicorn --port 9000 --host 0.0.0.0 --log-level debug app.webservice:app
