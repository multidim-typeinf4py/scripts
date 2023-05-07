FROM frolvlad/alpine-miniconda3

RUN touch /etc/inside-container

RUN apk add --no-cache curl && curl -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

RUN apk add --no-cache build-base 
RUN conda create --name monkeytype-base && conda run --name monkeytype-base pip install monkeytype
RUN conda install --name monkeytype-base poetry pytest pytest-timeout pip
