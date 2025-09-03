# CUDA 12.6 基础镜像 + Python 3.9（推荐官方 pytorch 镜像或 ubuntu 自构建）
FROM jittor/jittor:cuda12.2-cudnn8

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHON_VERSION=3.9.21
ENV TZ=Asia/Shanghai
WORKDIR /workspace

# 安装系统工具 & Python
RUN apt-get update && \
    apt-get install -y software-properties-common tzdata && \
    ln -fs /usr/share/zoneinfo/${TZ} /etc/localtime && \
    dpkg-reconfigure -f noninteractive tzdata && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.9 python3.9-dev python3-pip && \
    ln -s /usr/bin/python3.9 /usr/bin/python && \
    ln -s /usr/bin/pip3 /usr/bin/pip

# 拷贝项目代码
COPY . /workspace

# 安装依赖
COPY requirements.txt /workspace/requirements.txt
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# 初始化 Jittor
RUN python -m jittor.init.compiler && \
    python -m jittor.test.test_example

# 推理脚本加入权限
RUN chmod +x *.sh

# 默认进入 bash
CMD ["/bin/bash"]
