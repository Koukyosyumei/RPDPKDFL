# Reconstruct Private Data via Public Knowledge in Distillation-based Federated Learning

Code for Reconstruct Private Data via Public Knowledge in Distillation-based Federated Learning

## Install

```
./install.sh
```

## Dataset

- LFW

We use the masked LFW dataset of [DeepFace-EMD: Re-ranking Using Patch-wise Earth Mover’s Distance Improves Out-Of-Distribution Face Identification (2022)](https://arxiv.org/abs/2112.04016) by Hai Phan and Anh Nguyen. Our work utilses lfw-align-128.tar.gz and lfw-align-128-mask.tar.gz in their [Google Drive](https://drive.google.com/drive/folders/1hoyO7IWaIx2Km-pe4-Sn2D_uTFNLC7Ph?usp=sharing).

- LAG

We use LAG dataset of [Large Age-Gap Face Verification by Feature Injection in Deep Networks (2017)](http://www.ivl.disco.unimib.it/activities/large-age-gap-face-verification/) by Bianco and Simone. You can download the dataset with [this link](http://www.ivl.disco.unimib.it/wp-content/uploads/2016/09/LAGdataset_100.zip).

## How to run

### Prepare datasets

We assume that all data locates in `data` folder.

```
├── data
│   ├── LAGdataset_100.zip
│   ├── lfw-align-128-mask.tar.gz
│   └── lfw-align-128.tar.gz
```

- LFW

```
mkdir data/lag
unzip -qq data/LAGdataset_100.zip -d data/lag
```

- LAG

```
mkdir data/lfw
tar -zxf data/lfw-align-128.tar.gz -C data/lfw
tar -zxf data/lfw-align-128-mask.tar.gz -C data/lfw
```

### PTBI/TBI against FedKD

#### Usage

You can reproduce our experiments with [script/main.py](script/main.py).

```
usage: main.py [-h] [-t FEDKD_TYPE] [-d DATASET] [-a ATTACK_TYPE] [-c CLIENT_NUM] [-s SOFTMAX_TEMPREATURE]
               [-p PATH_TO_DATAFOLDER] [-o OUTPUT_FOLDER] [-b ABLATION_STUDY]

optional arguments:
  -h, --help            show this help message and exit
  -t FEDKD_TYPE, --fedkd_type FEDKD_TYPE
                        type of FedKD;
                            FedMD, FedGEMS, or FedGEMS
  -d DATASET, --dataset DATASET
                        type of dataset;
                            LAG or LFW
  -a ATTACK_TYPE, --attack_type ATTACK_TYPE
                        type of attack;
                            ptbi or tbi
  -c CLIENT_NUM, --client_num CLIENT_NUM
                        number of clients
  -s SOFTMAX_TEMPREATURE, --softmax_tempreature SOFTMAX_TEMPREATURE
                        tempreature $ au$
  -p PATH_TO_DATAFOLDER, --path_to_datafolder PATH_TO_DATAFOLDER
                        path to the data folder
  -o OUTPUT_FOLDER, --output_folder OUTPUT_FOLDER
                        path to the output folder
  -b ABLATION_STUDY, --ablation_study ABLATION_STUDY
                        type of ablation study;
                            0:normal (Q=p'_{c_i, j}+p'_{s, j}+lpha H(p'_s))
                            1:without entropy (Q=p'_{c_i, j}+p'_{s, j})
                            2:without p'_{s, j} (Q=p'_{c_i, j}+lpha H(p'_s))
                            3:without local logit (Q=p'_{s, j}+lpha H(p'_s))
```

#### Example

```
python script/main.py -t FedMD -d LAG -a ptbi -c 10 -s 3 -p ./data/lag -o path_to_output_folder -b 0
```

### Gradient-based attack against FedAVG

You can reproduce our experiments with [script/main_fedavg.py](script/main_fedavg.py).

#### Usage

```
usage: main_fedavg.py [-h] [-d DATASET] [-c CLIENT_NUM] [-p PATH_TO_DATAFOLDER] [-o OUTPUT_FOLDER]

optional arguments:
  -h, --help            show this help message and exit
  -d DATASET, --dataset DATASET
                        type of dataset;
                            LAG or LFW
  -c CLIENT_NUM, --client_num CLIENT_NUM
                        number of clients
  -p PATH_TO_DATAFOLDER, --path_to_datafolder PATH_TO_DATAFOLDER
                        path to the data folder
  -o OUTPUT_FOLDER, --output_folder OUTPUT_FOLDER
                        path to the output folder
```

#### Example

```
python script/main_fedavg.py -d LAG -c 10 -p ./data/lag -o path_to_output_folder
```