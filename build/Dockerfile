FROM nvidia/cuda:12.3.2-devel-ubuntu22.04

RUN apt update && \
	apt install -y curl python3.11 python3.11-distutils python3-pip wget git cmake libboost-all-dev

# making directory for copying mystem binary file
RUN cd ~ && mkdir local && mv local .local && cd .local && mkdir bin
RUN wget http://download.cdn.yandex.net/mystem/mystem-3.1-linux-64bit.tar.gz &&  \
    tar -zxvf mystem-3.1-linux-64bit.tar.gz && rm -rf mystem-3.1-linux-64bit.tar.gz && \
    mv mystem ~/.local/bin

COPY ../requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir --prefer-binary -r requirements.txt

ENV PROJECT_ROOT /app

ENV DATA_ROOT /data
ENV TEST_DATA_ROOT /test_data

RUN mkdir $DATA_ROOT # $PROJECT_ROOT

# dupicate original address http://nlpgrid.seas.upenn.edu/PPDB/rus/ppdb-1.0-s-lexical.gz
RUN mkdir $HOME/ppdb/ && wget https://storage.yandexcloud.net/varsey-backet/ppdb/ppdb-1.0-s-lexical.gz && \
    gunzip -c ppdb-1.0-s-lexical.gz > $HOME/ppdb/ppdb-1.0-s-lexical
WORKDIR $PROJECT_ROOT

# For lgbm GPU compatability
RUN git clone --recursive https://github.com/microsoft/LightGBM && cd LightGBM && mkdir build && cd build &&  \
    cmake -DUSE_GPU=1 .. && make -j4
RUN mkdir -p /etc/OpenCL/vendors
RUN echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd

# Download pretrained pytroch model from s3 storage
RUN mkdir $HOME/t-model/  && \
    wget -O $HOME/t-model/config.json  https://storage.yandexcloud.net/varsey-backet/artifacts/tt-model/config.json &&  \
    wget -O $HOME/t-model/training_args.bin  https://storage.yandexcloud.net/varsey-backet/artifacts/tt-model/training_args.bin &&  \
    wget -O $HOME/t-model/pytorch_model.bin  https://storage.yandexcloud.net/varsey-backet/artifacts/tt-model/pytroch_model.bin

# TO-DO move to pip install reqs stage
RUN pip install boto3

COPY lib/ $PROJECT_ROOT/lib/
RUN python3 -c "from lib.src.transformer_trainer import TransformerTrainer; TransformerTrainer()"
RUN python3 -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('averaged_perceptron_tagger')"

COPY main.py $PROJECT_ROOT

# CMD python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
CMD python3 -m uvicorn main:app