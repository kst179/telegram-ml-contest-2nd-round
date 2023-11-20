FROM debian:10

WORKDIR /app

RUN apt-get update && \
    apt-get install -y cmake g++

COPY ./src /app/src
COPY ./CMakeLists.txt /app/
COPY ./input.txt /app/

ARG BUILD_TYPE=Release
ARG EMBED_WEIGHTS=ON
ARG USE_AVX_EXP=ON

RUN mkdir /app/buster_build
RUN cmake -B /app/buster_build -DCMAKE_BUILD_TYPE=${BUILD_TYPE} -DEMBED_WEIGHTS=${EMBED_WEIGHTS} -DUSE_AVX_EXP=${USE_AVX_EXP}

RUN cmake --build /app/buster_build
