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



### Gradient-based attack against FedAVG