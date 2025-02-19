FROM nvcr.io/nvidia/pytorch:25.01-py3

RUN apt-get update && apt-get upgrade -y && apt-get install ca-certificates lsb-release wget python3-pip neovim autoconf build-essential gdb software-properties-common curl unzip cmake gzip protobuf-compiler libtool zstd liblz4-dev lz4 -y

RUN wget https://apache.jfrog.io/artifactory/arrow/$(lsb_release --id --short | tr 'A-Z' 'a-z')/apache-arrow-apt-source-latest-$(lsb_release --codename --short).deb
RUN apt install -y -V ./apache-arrow-apt-source-latest-$(lsb_release --codename --short).deb
RUN apt update
RUN apt install -y -V  libparquet-glib-dev libparquet-dev libarrow-dataset-glib-dev libarrow-dataset-dev libarrow-glib-dev libarrow-dev

RUN pip install pip==24.*

# If you encounter pyarrow issues, ensure the version here matches the version downloaded above!!
RUN pip install tqdm loguru psutil numpy==1.26.4 dill datasets transformers pyarrow==19.*  xxhash xopen scipy tenacity
RUN pip install duckdb polars==1.15 pillow pybind11 pytest flake8 mypy pylint autopep8 isort black tensorboard tiktoken blobfile tabulate wandb torchdata>=0.8.0 tomli>=1.1.0 dacite pyyaml packaging safetensors sentencepiece jupyter seaborn webdataset lz4  git+https://github.com/tmbdev/webdataset.git@v0.2.107 mosaicml-streaming grain
RUN pip install lm_eval typer # for evaluation

# Test torch nightly
RUN pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124

RUN git clone --recurse-submodules -b v1.64.3 --depth 1 --shallow-submodules https://github.com/grpc/grpc && \
    cd grpc && mkdir -p cmake/build && cd cmake/build && \
    cmake -DgRPC_PROTOBUF_PROVIDER=module -DABSL_ENABLE_INSTALL=On -DgRPC_BUILD_CSHARP_EXT=Off -DABSL_BUILD_TESTING=Off -DgRPC_INSTALL=ON -DgRPC_BUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release ../.. && \
    make -j64 && make install && cd ../../

RUN bash -c "cp /usr/local/lib/libutf8* /usr/lib"

## For nanotron
RUN pip uninstall -y ninja && pip install ninja
RUN MAX_JOBS=12 numactl --membind=0-3 pip install  flash-attn --no-build-isolation