## AppUsage2Vec

This is Pytorch implementation for the paper:

> Zhao, S., Luo, Z., Jiang, Z., Wang, H., Xu, F., Li, S., ... & Pan, G. (2019, April). Appusage2vec: Modeling smartphone app usage for prediction. In 2019 IEEE 35th International Conference on Data Engineering (ICDE) (pp. 1322-1333). IEEE.

### Introduction

### Environment Requirement

*Docker
```
docker build -t appusage2vec .
docker run --rm -it --gpus all -p 8888:8888 -v {your_path}/AppUsage2Vec:/AppUsage2Vec appusage2vec /bin/bash
```

```
jupyter lab --allow-root --ip=0.0.0.0
```
and go to 127.0.0.1 for using jupyter lab in the container

### Dataset

### How to run
```
python main.py
```
