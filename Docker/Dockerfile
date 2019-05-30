FROM ubuntu:latest

RUN apt update
RUN apt install -y	build-essential
RUN apt install -y  nvidia-cuda-toolkit
RUN apt install -y	cmake
RUN apt install -y  g++-8
RUN apt install -y	git
RUN apt -y autoremove

RUN git clone https://github.com/StuartRiffle/Jaglavak

RUN mkdir Jaglavak/build
WORKDIR /Jaglavak/build
RUN cmake ..
RUN make




