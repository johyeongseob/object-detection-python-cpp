# syntax=docker/dockerfile:1

FROM ubuntu:22.04 AS builder

ARG DEBIAN_FRONTEND=noninteractive
ARG ONNXRUNTIME_VERSION=1.23.2

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        libopencv-dev \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY cpp/ ./cpp/

RUN archive="onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz" \
    && url="https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/${archive}" \
    && mkdir -p cpp/third_party \
    && curl --fail --location --show-error --silent \
        "${url}" \
        --output "/tmp/${archive}" \
    && tar -xzf "/tmp/${archive}" -C cpp/third_party \
    && rm "/tmp/${archive}"

RUN cmake \
        -S cpp \
        -B cpp/build \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_TESTING=ON \
    && cmake --build cpp/build --parallel \
    && ctest --test-dir cpp/build --output-on-failure

FROM ubuntu:22.04 AS runtime

ARG DEBIAN_FRONTEND=noninteractive
ARG ONNXRUNTIME_VERSION=1.23.2

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        ca-certificates \
        libopencv-core4.5d \
        libopencv-dnn4.5d \
        libopencv-imgcodecs4.5d \
        libopencv-imgproc4.5d \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder \
    /app/cpp/third_party/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}/lib/libonnxruntime.so* \
    /opt/onnxruntime/lib/
COPY --from=builder /app/cpp/build/detect_person_image /usr/local/bin/
COPY --from=builder /app/cpp/build/evaluate_coco_person /usr/local/bin/
COPY --from=builder /app/cpp/build/person_detection_server /usr/local/bin/
COPY --from=builder /app/cpp/build/verify_yolo11n_onnx /usr/local/bin/

ENV LD_LIBRARY_PATH=/opt/onnxruntime/lib
WORKDIR /workspace
EXPOSE 8080

CMD ["person_detection_server"]
