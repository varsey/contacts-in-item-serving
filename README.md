# Project for demonstration ML-model serving and retraining

* sudo docker build -t contact-in-item:docker -f build/Dockerfile .
* sudo docker run -it -p 8000:8000 --gpus all eb36384a3c32 /bin/bash

OR 

* sudo docker-compose -f build/docker-compose.yaml up
