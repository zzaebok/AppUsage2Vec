FROM pytorch/pytorch:1.5.1-cuda10.1-cudnn7-runtime
MAINTAINER Jaebok Lee "jaebok111@naver.com"

RUN apt-get -y update

WORKDIR /AppUsage2Vec
COPY . /AppUsage2Vec

RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 8888

CMD python main.py
