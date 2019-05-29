FROM ubuntu:18.04

RUN apt update
RUN apt install -y	git
RUN apt install -y	cmake
RUN apt install -y	build-essential
RUN apt install -y	nvidia-cuda-dev

RUN git clone https://github.com/StuartRiffle/Jaglavak

RUN mkdir Jaglavak/build
WORKDIR /Jaglavak/build

RUN cmake ..
RUN make




