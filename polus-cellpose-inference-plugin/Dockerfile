FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime

COPY VERSION /

ARG EXEC_DIR="/opt/executables"
ARG DATA_DIR="/data"
ARG HOME=/
ENV NUMBA_CACHE_DIR=${EXEC_DIR}/numba_cache

RUN pip install --upgrade pip --no-cache-dir

RUN mkdir -p ${EXEC_DIR} \
    && mkdir -p ${DATA_DIR}/inputs \
    && mkdir ${DATA_DIR}/outputs

COPY src ${EXEC_DIR}/
WORKDIR ${EXEC_DIR}

RUN pip3 install -r ${EXEC_DIR}/requirements.txt --no-cache-dir

RUN python -c "import cellpose.models"

RUN chmod -R 777 ${EXEC_DIR}/numba_cache && \
    chmod -R 777 /.cellpose

ENTRYPOINT ["python3", "main.py"]