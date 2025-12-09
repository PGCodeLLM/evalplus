# Better use newer Python as generated code can use new features
FROM python:3.11-slim

# install git and c++ (required by cirronlib.cpp)
RUN apt-get update && apt-get install -y git g++

# upgrade to latest pip
RUN pip install --upgrade pip

COPY . /evalplus

# Set a version for setuptools-scm since we don't have .git metadata
ENV SETUPTOOLS_SCM_PRETEND_VERSION_FOR_EVALPLUS=0.3.1

RUN cd /evalplus && pip install ".[perf]"

ENV REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
COPY certs /usr/local/share/ca-certificates
RUN update-ca-certificates

# Pre-install the dataset
RUN python3 -c "from evalplus.data import *; get_human_eval_plus(); get_mbpp_plus(); get_evalperf_data()"

WORKDIR /app

CMD ["bash"]
