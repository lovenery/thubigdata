FROM ubuntu:18.04

USER root
WORKDIR /root

COPY . /root

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
 && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade setuptools
RUN pip3 install -r requirements.txt

