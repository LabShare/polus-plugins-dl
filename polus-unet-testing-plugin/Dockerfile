
FROM labshare/polus-bfio-util:2.1.8

COPY VERSION /
		
ARG EXEC_DIR="/opt/executables"
ARG DATA_DIR="/data"

RUN mkdir -p ${EXEC_DIR} \
    && mkdir -p ${DATA_DIR}/inputs \
    && mkdir ${DATA_DIR}/outputs

COPY src ${EXEC_DIR}/
WORKDIR ${EXEC_DIR}

RUN pip3 install -r ${EXEC_DIR}/requirements.txt --no-cache-dir

RUN apt-get update
RUN apt-get install -y software-properties-common && \
    apt-get install -y gnupg && \
    apt-get install -y wget && \
    apt install -y build-essential && \ 
    apt-get install -y doxygen libssl-dev git

RUN version=3.15 && build=1 && \
    mkdir ~/temp && cd ~/temp && \
    wget https://cmake.org/files/v$version/cmake-$version.$build.tar.gz && \
    tar -xzvf cmake-$version.$build.tar.gz && \
    cd cmake-$version.$build/ && \
    ./bootstrap && make -j$(nproc) && make install && cmake --version

# install dependencies for caffe binary
RUN apt-get install -y unzip build-essential libboost-system-dev libboost-thread-dev libboost-filesystem-dev libprotobuf-dev protobuf-compiler libhdf5-serial-dev libatlas-base-dev libgoogle-glog-dev 

# # install caffe binary
RUN mkdir ${EXEC_DIR}/unetuser && \
    cd ${EXEC_DIR}/unetuser && \
    git clone https://github.com/BVLC/caffe.git && \
    cd caffe &&\
    git checkout 99bd99795dcdf0b1d3086a8d67ab1782a8a08383 && \
    wget https://lmb.informatik.uni-freiburg.de/lmbsoft/unet/caffe_unet_99bd99_20190109.patch && \
    git apply caffe_unet_99bd99_20190109.patch && \
    mkdir x86_64 && \
    cd x86_64 && \
    cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=${EXEC_DIR}/unetuser/u-net -DUSE_OPENCV=OFF -DUSE_LEVELDB=OFF -DUSE_LMDB=OFF -DBUILD_python=OFF -DBUILD_python_layer=OFF -DCPU_ONLY=ON .. && \
    make -j2 install

    
ENV PATH=$PATH:${EXEC_DIR}/unetuser/u-net/bin
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${EXEC_DIR}/unetuser/u-net/lib

ENTRYPOINT ["python3", "main.py"]
