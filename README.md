# AppUsage2Vec

This is Pytorch implementation for the paper:

> Zhao, S., Luo, Z., Jiang, Z., Wang, H., Xu, F., Li, S., ... & Pan, G. (2019, April). Appusage2vec: Modeling smartphone app usage for prediction. In 2019 IEEE 35th International Conference on Data Engineering (ICDE) (pp. 1322-1333). IEEE.

## Introduction

AppUsage2Vec is a novel framework for app usage prediction.
This model exploited 'App Attention', 'Dual DNN', 'Temporal Context', and 'Top k Based Optimization' structures.

## Environment Requirement

The code has been tested running under Python 3.7.7.
Look at the `requirements.txt` for more detail

- torch==1.5.1
- scikit-learn==0.23.2
- jupyter-lab==2.2.6
- pandas==1.1.3
- matplotlib==3.3.4

- CUDA 10.1
- CUDNN 7

## Docker

If you want to run the code by using docker, follow instructions below.

* Docker build

```cmd
docker build -t appusage2vec .
docker run --rm -it --gpus all -p 8888:8888 -v {your_path}/AppUsage2Vec:/AppUsage2Vec appusage2vec /bin/bash
```

* Jupyter lab in container

```cmd
jupyter lab --allow-root --ip=0.0.0.0
```
And copy the url in the terminal and paste it on the web browser to use jupyterlab in the container.

## Dataset

[Tsinghua App Usage Dataset](http://fi.ee.tsinghua.edu.cn/appusage/)

```
@article{yu2018smartphone,
    title={Smartphone app usage prediction using points of interest},
    author={Yu, Donghan and Li, Yong and Xu, Fengli and Zhang, Pengyu and Kostakos, Vassilis},
    journal={Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies},
    volume={1},
    number={4},
    pages={174},
    year={2018},
    publisher={ACM}}
```
Extract `App_usage_trace.txt` and put in in the `data` directory.

## Preprocessing

Run `preprocessing.ipynb`.
If you want to change sequence length, change `seq_length` in 4th cell.

## How to run

```
python main.py
```

## Result

Please note that the result is not fine-trained well. (Just used default arguments in `main.py`)

<img src=result.png width=600>