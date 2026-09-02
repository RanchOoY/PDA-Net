# PDA-Net
# PDA-Net: Prior-Guided Degradation-Aware Network for Low-Light Image Enhancement

<div align="center">

  

## Description

This is the PyTorch version of PNA-Net.

## Experiment

### 1. Create Environment

- Make Conda Environment

```bash
conda create -n PDA_Torch python=3.9 -y
conda activate PDA_Torch
```

- Install Dependencies

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.8 -c pytorch -c nvidia

pip install matplotlib scikit-learn scikit-image opencv-python yacs joblib natsort h5py tqdm tensorboard

pip install einops gdown addict future lmdb numpy pyyaml requests scipy yapf lpips thop timm torchmetrics pytorch_msssim
```

### 2. Prepare Datasets

Download the LOLv1 and LOLv2 datasets:

LOLv1 - [Google Drive](https://drive.google.com/file/d/1vhJg75hIpYvsmryyaxdygAWeHuiY_HWu/view?usp=sharing)

LOLv2 - [Google Drive](https://drive.google.com/file/d/1OMfP6Ks2QKJcru1wS2eP629PgvKqF2Tw/view?usp=sharing)

**Note:** Under the main directory, create a folder called ```data``` and place the dataset folders inside it.

<details>
  <summary>
  <b>Datasets should be organized as follows:</b>
  </summary>


  ```
  |--data   
  |    |--LOLv1
  |    |    |--Train
  |    |    |    |--input
  |    |    |    |     ...
  |    |    |    |--target
  |    |    |    |     ...
  |    |    |--Test
  |    |    |    |--input
  |    |    |    |     ...
  |    |    |    |--target
  |    |    |    |     ...
  |    |--LOLv2
  |    |    |--Real_captured
  |    |    |    |--Train
  |    |    |    |    |--Low
  |    |    |    |    |     ...
  |    |    |    |    |--Normal
  |    |    |    |    |     ...
  |    |    |    |--Test
  |    |    |    |    |--Low
  |    |    |    |    |     ...
  |    |    |    |    |--Normal
  |    |    |    |    |     ...
  |    |    |--Synthetic
  |    |    |    |--Train
  |    |    |    |    |--Low
  |    |    |    |    |    ...
  |    |    |    |    |--Normal
  |    |    |    |    |    ...
  |    |    |    |--Test
  |    |    |    |    |--Low
  |    |    |    |    |    ...
  |    |    |    |    |--Normal
  |    |    |    |    |    ...
  ```

</details>

**Note:** ```data``` directory should be placed under the ```PyTorch``` implementation folder.

### 3. Test

You can test the model using the following commands. 

```bash
python test.py
```

**Note:** Please modify the dataset paths in ```test.py``` as per your requirements.

### 4. Compute Complexity

You can test the model complexity (FLOPS/Params) using the following command:

```bash
python macs.py
```

### 5. Train

You can train the model using the following command:

```bash
python train.py
```

**Note:** Please modify the dataset paths in ```train.py``` as per your requirements.

