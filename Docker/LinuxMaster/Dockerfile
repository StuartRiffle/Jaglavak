FROM ubuntu:latest

RUN apt update
RUN apt install -y	build-essential cmake git
RUN apt install -y  nvidia-cuda-toolkit
RUN apt -y autoremove

ENTRYPOINT \
    git clone https://github.com/StuartRiffle/Jaglavak &&\
    cd Jaglavak &&\
    git checkout master &&\
    mkdir Build &&\
    cd Build &&\
    cmake -DCMAKE_BUILD_TYPE=Release .. &&\
    make &&\
    ./Jaglavak





