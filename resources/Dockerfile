FROM frolvlad/alpine-miniconda3

RUN touch /etc/inside-container

RUN apk update && apk add --no-cache bash curl && curl -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

RUN apk add --no-cache build-base
RUN conda update -n base -c defaults conda --repodata-fn=repodata.json

RUN conda create --name monkeytype-base python=3.11 && conda run --name monkeytype-base \
    pip install monkeytype
RUN conda install --name monkeytype-base poetry pytest pytest-timeout pip

# Ensure we do not have busybox versions of these programs for run.sh
RUN apk add --no-cache grep

# For debugging
RUN apk add --no-cache vim

# Dependencies discovered from failing installations on BetterTypes4Py
RUN apk add --no-cache alpine-sdk zlib-dev jpeg-dev

COPY run.sh /