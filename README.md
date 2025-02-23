# miccai25-struct-da-code

Python/PyTorch codes for our paper (paper id: 1978) submitted to MICCAI25.

## 1. Python environment

The code is tested with Python 3.11.9 and CUDA 12.4.1.

Firstly create a python virtual environment:

```bash
$ python3 -m venv /path/to/env/folder/env1
$ source /path/to/env/folder/env1/bin/activate
```

Update pip:

```bash
(env1) $ pip install -U pip
```

Install packages using ```requirements.txt```:

```bash
(env1) $ pip install -r requirements.txt
```

## 2. Prepare dataset

Our paper utilizes four datasets; however, for simplicity, we provide dataset preparation code only for the UNC 3T-7T paired dataset. Other datasets can be prepared using similar code with minor modifications to the data loading process.

Step 1. Download the UNC paired 3T-7T dataset using their official link: [paper](https://www.nature.com/articles/s41597-025-04586-9), [host page](https://springernature.figshare.com/articles/dataset/UNC_Paired_3T-7T_Dataset/23706033), [download link](https://springernature.figshare.com/ndownloader/files/41605158)

Step 2. Run data pre-processing script with Python.

```bash
python preprocess_unc_dataset.py --zip_path /path/to/unc/dataset.zip --dataset_name unc --dataset_root ./dataset
```

--zip_path: UNC paired 3T-7T dataset zip file path

--dataset_name: dataset name for output npy folder

--dataset_root: root directory for preprocessed (i.e., npy) data

## 3. Training

Run training script with the configure file:

```bash
python train.py --config_path ./configs/unc-config.yaml
```

Please make sure the ```data_root_dir``` is set as the correct path.
