# miccai25-struct-da-code

Python/PyTorch codes for my paper submitted to MICCAI25.

## 0. Prepare dataset

Our paper utilizes four datasets; however, for simplicity, we provide dataset preparation code only for the UNC 3T-7T paired dataset. Other datasets can be prepared using similar code with minor modifications to the data loading process.

- 0.1 Download the UNC paired 3T-7T dataset using their official link: [paper](https://www.nature.com/articles/s41597-025-04586-9), [host page](https://springernature.figshare.com/articles/dataset/UNC_Paired_3T-7T_Dataset/23706033), [download link](https://springernature.figshare.com/ndownloader/files/41605158)
- 0.2 Run data pre-processing script with Python.
"""python
python run.py
"""
