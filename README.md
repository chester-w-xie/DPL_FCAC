# [Few-shot class-incremental audio classification via discriminative prototype learning](https://www.sciencedirect.com/science/article/pii/S0957417423005468?via%3Dihub)

Official PyTorch implementation of **DPL**, from the following paper:

[Few-shot class-incremental audio classification via discriminative prototype learning](https://www.sciencedirect.com/science/article/pii/S0957417423005468?via%3Dihub). Expert Systems with Applications 2023.\
[Wei Xie](https://scholar.google.com/citations?hl=en&user=bOUL1r4AAAAJ), [Yanxiong Li](https://scholar.google.com/citations?user=ywDuJjEAAAAJ&hl=en), [Qianhua He](https://scholar.google.com/citations?user=xgI45kMAAAAJ&hl=en) and Wenchang Cao\
School of Electronic & Information Engineering, South China University of Technology


--- 


### The problem of Few-shot class-incremental audio classification
<div align="center">
  <img src='figs/FCAC.png' width="100%" height="100%" alt=""/>
</div>







### The method of discriminative prototype learning

<div align="center">
  <img src='figs/DPL.png' width="100%" height="100%" alt=""/>
</div>



## 0. Requirements

[Conda]( https://conda.io/projects/conda/en/latest/user-guide/install/index.html?highlight=conda ) should be installed on the system.

* Install [Anaconda](https://www.anaconda.com/).

* Run the install dependencies script:
```bash
conda env create -f environment.yml
```
This creates conda environment ```FCAC``` with all the dependencies.

## Datasets

Please follow the instructions [here](https://github.com/chester-w-xie/FCAC_datasets) to prepare the NSynth-100 and FSC-89 datasets.

## Usage

## Main results

<div align="left">
  <img src='figs/Main_result.png' width="80%" height="80%" alt=""/>
</div>


## Acknowledgement
This repository is built using 
## License
This project is released under the MIT license. Please see the [LICENSE](LICENSE) file for more information.

## Citation
If you find this repository helpful, please consider citing:
```
@article{XIE2023120044,
title = {Few-shot class-incremental audio classification via discriminative prototype learning},
journal = {Expert Systems with Applications},
pages = {120044},
year = {2023},
issn = {0957-4174},
doi = {https://doi.org/10.1016/j.eswa.2023.120044},
url = {https://www.sciencedirect.com/science/article/pii/S0957417423005468},
author = {Wei Xie and Yanxiong Li and Qianhua He and Wenchang Cao},
}
```
