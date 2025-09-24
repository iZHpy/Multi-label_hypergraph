FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04 
 # Do not need use cuda, so just runtime image
 # if need compile CUDA extensions, use -devel- tag

# install micromamba (lightweight conda)
ARG MAMBA_VERSION=1.5.8
RUN apt-get update && apt-get install -y --no-install-recommends curl bzip2 ca-certificates tini && \
    curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/${MAMBA_VERSION} \
    | tar -xvj -C /usr/local/bin bin/micromamba --strip-components=1 && \
    rm -rf /var/lib/apt/lists/*

ARG ENV_NAME=hypergraph
ENV MAMBA_ROOT_PREFIX=/opt/conda
RUN mkdir -p $MAMBA_ROOT_PREFIX

WORKDIR /opt/app
COPY environment.yml /opt/app/environment.yml

# 先按 yml 创建环境（若 torch/pyg 冲突，可注释掉，改为下一步手工装）
RUN micromamba create -y -n $ENV_NAME -f environment.yml && \
    micromamba clean -a -y

# 针对 torch==1.13.1 推荐用官方 cu117 wheel（更稳），如需覆盖：
# RUN micromamba run -n $ENV_NAME pip install \
#   torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 \
#   --extra-index-url https://download.pytorch.org/whl/cu117 && \
#   micromamba run -n $ENV_NAME pip install torch-geometric==2.6.0

ENV PATH=$MAMBA_ROOT_PREFIX/envs/$ENV_NAME/bin:$PATH

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "-V"]
