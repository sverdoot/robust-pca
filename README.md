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

### Extracting activities on video surveillance
Example of solving Principle Component Pusuit using Augmented Lagrangian Multipliers method 
for decomposing video frames (1-st row) into static background (2-nd row) and dynamic foreground (3-rd row)
<p align="center"><img src="figs/pcp_mall" width="700" /></p>

### Removing shadows from face images
Example of solving Principle Component Pusuit using Augmented Lagrangian Multipliers method 
for decomposing face images (1-st row) into low-rank face approximation (2-nd row) and sparse shadows (3-rd row)
<p align="center"><img src="figs/pcp_yale" width="700" /></p>

## Experiments accomplished
We compare different tasks for removing shadows from face images and extracting background/foreground from video frames:
1. PCP via ALM
2. Stable PCP via alternating directions
3. PCP with compressed data via alternating directions
4. PCP via Iterated Robust CUR

We also demonstrate RPCA with compressed data by applying it to compressed YALE face dataset.

## Repository structure
- ```notebooks/``` — contains experiments in form of jupyter notebooks \
    ```├── demo_mall.ipynb``` — demonstration of extracting activities from video surveillance \
    ```├── demo_yale.ipynb``` — demonstration of removing shadows from face images\
    ```├── RPCA with compressed data.ipynb``` — demonstration solving RPCA with compressed images\
- ```data/``` — folder for relevant datasets
- ```robustpca/``` — related source code with implementation of algorithms
- ```figs/``` — pictures for the [results](#results) part
- ```runs/``` — bash scripts to run experiments

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

## Results
### Extracting activities on video surveillance

#### PCP via ALM
<p align="center"><img src="figs/pcp_mall" width="700" /></p>

---
#### Stable PCP via alternating directions
<p align="center"><img src="figs/stable_pcp_mall" width="700" /></p>

---
#### PCP via IRCUR
<p align="center"><img src="figs/ircur_mall" width="700" /></p>

---

### Removing shadows from face images

#### PCP via ALM
<p align="center"><img src="figs/pcp_yale" width="700" /></p>

---
#### Stable PCP via alternating directions
<p align="center"><img src="figs/stable_pcp_yale" width="700" /></p>

---
#### PCP via IRCUR
<p align="center"><img src="figs/ircur_yale" width="700" /></p>

---

