# Project for demonstration ML-model serving and retraining

Classification web-service in cluster to detect if personal information is present in classified ads, prediction and retraining pipelines 
Technology stack: numpy, pandas, [sklearn linreg, lightgbm, pytroch], dask, fastapi, uvicorn, boto3, k8s s3, github actions, docker hub registry, promitheus, grafana)

![Screenshot from 2024-03-06 20-05-54](https://github.com/varsey/contacts-in-item-serving/assets/21172646/16ef1dae-f59f-4cf2-a136-b8b1195e3985)

How to run locally:
* sudo docker build -t contact-in-item:docker -f build/Dockerfile .
* sudo docker run -it -p 8000:8000 --gpus all <docker-image-id> /bin/bash

OR 

* sudo docker-compose -f build/docker-compose.yaml up

## Slides
Slide deck is [here](https://docs.google.com/presentation/d/1FyGRcOFEhQKE6yhlNu-c09ntF5k9UnRn9u648EhBhXo/edit?usp=sharing)
