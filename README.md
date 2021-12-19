# Robust PCA

- [Robust PCA](#robust-pca)
  - [Intro](#intro)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Citing](#citing)

## Intro

This repository provides implementation of the following algorithms:

* PCP via ALM method: [Robust Principal Component Analysis?, Candes et.al., 2009](cite)

* Stable PCP via alternating directions (not accelerated): [Stable Principal Component Persuit,  Zhou et. al., 2010](cite)

* PCP with compressed data via alternating directions: [Robust PCA with compressed data, Ha and Barber, 2011](https://proceedings.neurips.cc/paper/2015/hash/c44e503833b64e9f27197a484f4257c0-Abstract.html)

* Iterated Robust CUR: [Rapid Robust Principal Component Analysis: CUR Accelerated Inexact Low Rank Estimation, Cai et. al, 2021](https://arxiv.org/abs/2010.07422)


## Installation

Setup the environment and install package:

```bash
conda create -n robustpca python=3.9
conda activate robustpca
pip install poetry
poetry install
chmod +x **.sh
```

Download data for experiments:

```bash
wget -nc http://vision.ucsd.edu/extyaleb/CroppedYaleBZip/CroppedYale.zip -P data
unzip data/CroopedYale.zip -d data

wget https://personal.ie.cuhk.edu.hk/\~ccloy/files/datasets/mall_dataset.zip -P data
unzip data/mall_dataset.zip -d data
```

## Usage

## Citing

