# A Neuro-explicit Model For Single-channel EEG-based Sleep Staging
By Jakob Scheytt

This repository provides the full source code of the Bachelor's Thesis "A Neuro-explicit Model For Single-channel EEG-based Sleep Staging" by Jakob Scheytt. 
The thesis was written at the Technische Universität Berlin, Faculty IV, Department of Electronic Systems of Medical Engineering

To reproduce the results of SleePyCo, please follow the instructions on the bottom of this page.

To reproduce the results achieved by the hybrid model, follow the steps below.

1. Follow steps 1-4 of SleePyCo's instructions to set up the environment and download the Sleep-EDF dataset. It is sufficient to download the Sleep-EDF-2018 dataset.
Note: if you want to use different datasets, you will need to adapt the code in all below-mentioned functions accordingly.

2. Run the function "Calc_P_Matr" with the following argument:
"--config "configs/SleePyCo-Transformer_SL-10_numScales-3_Sleep-EDF-2018_freezefinetune.json" --gpu  GPU_IDs"
You need to create three different P matrices, for the training, test, and validation set. Therefore, you need to run the
function three times and adapt the parameter "self.set" and the name of the output file accordingly.
- If you want to run the function on the HPC, you can use the commented out code to determine the right gpu ids.

3. Run the function "Calc_Trans_Matr_Static.py" to calculate the transition matrix from train and validation labels.

4. If you have adapted the output name, you will have to adapt the functions "load_P_Matrix" and "load_Transition_Matrix" 
in the HMM_utils.py file. If not, you can now run the hybrid model by running the function "DNN_and_Vit.py" which will call
the function "Viterbi_Algorithm""









# SleePyCo

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/sleepyco-automatic-sleep-scoring-with-feature/sleep-stage-detection-on-sleep-edf)](https://paperswithcode.com/sota/sleep-stage-detection-on-sleep-edf?p=sleepyco-automatic-sleep-scoring-with-feature) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/sleepyco-automatic-sleep-scoring-with-feature/sleep-stage-detection-on-sleep-edfx)](https://paperswithcode.com/sota/sleep-stage-detection-on-sleep-edfx?p=sleepyco-automatic-sleep-scoring-with-feature) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/sleepyco-automatic-sleep-scoring-with-feature/sleep-stage-detection-on-mass-single-channel)](https://paperswithcode.com/sota/sleep-stage-detection-on-mass-single-channel?p=sleepyco-automatic-sleep-scoring-with-feature) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/sleepyco-automatic-sleep-scoring-with-feature/sleep-stage-detection-on-physionet-challenge)](https://paperswithcode.com/sota/sleep-stage-detection-on-physionet-challenge?p=sleepyco-automatic-sleep-scoring-with-feature) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/sleepyco-automatic-sleep-scoring-with-feature/sleep-stage-detection-on-shhs-single-channel)](https://paperswithcode.com/sota/sleep-stage-detection-on-shhs-single-channel?p=sleepyco-automatic-sleep-scoring-with-feature)


By Seongju Lee, Yeonguk Yu, Seunghyeok Back, Hogeon Seo, and Kyoobin Lee

This repo is the official implementation of "***SleePyCo: Automatic Sleep Scoring with Feature Pyramid and Contrastive Learning***", submitted to Expert Systems With Applications.

[[Arxiv](https://arxiv.org/abs/2209.09452)]

## Model Architecture
![model](./figures/model.png)

## Training Framework
![framework](./figures/framework.png)

## Updates & TODO Lists
- [X] (2023.03.03) Official repository of SleePyCo is released
- [X] Script for preprocessing Sleep-EDF
- [X] Config files for training from scratch
- [ ] Config files for ablation studies
- [ ] Scripts for preprocessing MASS, Physio2018, SHHS


## Getting Started

### Environment Setup

Trained and evaluated on NVIDIA GeForce RTX 3090 with python 3.8.5.

1. Set up a python environment
```
conda create -n sleepyco python=3.8.5
conda activate sleepyco
```

2. Install PyTorch with compatible version to your develop env from [PyTorch official website](https://pytorch.org/).

3. Install remaining libraries using the following command.
```
pip install -r requirements.txt
```

### Dataset Preparation
#### Sleep-EDF dataset
1. Download `Sleep-EDF-201X` dataset via following command. (`X` will be `3` or `8`)
```
cd ./dset/Sleep-EDF-201X
python download_sleep-edf-201X.py
```

2. Check the directory structure as follows
```
./dset/
└── Sleep-EDF-201X/
    └── edf/
        ├── SC4001E0-PSG.edf
        ├── SC4001EC-Hypnogram.edf
        ├── SC4002E0-PSG.edf
        ├── SC4002EC-Hypnogram.edf
        ├── ...
```

3. Preprocess `.edf` files into `.npz`.
```
python prepare_sleep-edf-201X.py
```

4. Check the directory structure as follows
```
./dset/
└── Sleep-EDF-201X/
    ├── edf/
    │   ├── SC4001E0-PSG.edf
    │   ├── SC4001EC-Hypnogram.edf
    │   ├── SC4002E0-PSG.edf
    │   ├── SC4002EC-Hypnogram.edf
    │   ├── ...
    │
    └── npz/
        ├── SC4001E0-PSG.npz
        ├── SC4002E0-PSG.npz
        ├── ...
```

## Train & Evaluation (SleePyCo Training Framework)
### Contrastive Representation Learning
```
python train_crl.py --config configs/SleePyCo-Transformer_SL-01_numScales-1_Sleep-EDF-2013_pretrain.json --gpu 0
python train_crl.py --config configs/SleePyCo-Transformer_SL-01_numScales-1_Sleep-EDF-2018_pretrain.json 
```
When one GeForce RTX 3090 GPU is used, it may requires 21.5 GB of GPU memory.

### Multiscale Temporal Context Learning
```
python train_mtcl.py --config configs/SleePyCo-Transformer_SL-10_numScales-3_Sleep-EDF-2013_freezefinetune.json --gpu 0
python train_mtcl.py --config configs/SleePyCo-Transformer_SL-10_numScales-3_Sleep-EDF-2018_freezefinetune.json --gpu 0
```3
When two GeForce RTX 3090 GPU is used, it may requires 16.7 GB of GPU memory each.

## Train & Evaluation (From Scratch)
```
python train_mtcl.py --config configs/SleePyCo-Transformer_SL-10_numScales-3_Sleep-EDF-2013_scratch.json --gpu 0
python train_mtcl.py --config configs/SleePyCo-Transformer_SL-10_numScales-3_Sleep-EDF-2018_scratch.json --gpu 0
```

## Main Results
|   **Dataset**  | **Subset** | **Channel** | **ACC** | **MF1** | **Kappa** | **W** | **N1** | **N2** | **N3** | **REM** | **Checkpoints** |
|:--------------:|:----------:|:-----------:|:-------:|:-------:|:---------:|:-----:|:------:|:------:|:------:|:-------:|:---------------:|
| Sleep-EDF-2013 |     SC     |    Fpz-Cz   |   86.8  |   81.2  |   0.820   |  91.5 |  50.0  |  89.4  |  89.0  |   86.3  | [Link](https://drive.google.com/file/d/1oUs8S9dVwmTJi9t9zh7msmJT_B28OpbP/view?usp=sharing) |
| Sleep-EDF-2018 |     SC     |    Fpz-Cz   |   84.6  |   79.0  |   0.787   |  93.5 |  50.4  |  86.5  |  80.5  |   84.2  | [Link](https://drive.google.com/file/d/1RdWl9AUMkFlNwUE2qxx3v5XcL3Exs0Pk/view?usp=sharing) |
|      MASS      |   SS1-SS5  |    C4-A1    |   86.8  |   82.5  |   0.811   |  89.2 |  60.1  |  90.4  |  83.8  |   89.1  | [Link](https://drive.google.com/file/d/16kPPhW04g5swGQeOJs8aRJOI13wSEKhI/view?usp=sharing)                 |
|   Physio2018   |      -     |    C3-A2    |   80.9  |   78.9  |   0.737   |  84.2 |  59.3  |  85.3  |  79.4  |   86.3  | [Link](https://drive.google.com/file/d/1r4NXeSzmP5rp_WTTGxiwHLGzknjPV8PT/view?usp=sharing) |
|      SHHS      |   shhs-1   |    C4-A1    |   87.9  |   80.7  |   0.830   |  92.6 |  49.2  |  88.5  |  84.5  |   88.6  | [Link](https://drive.google.com/file/d/1FwjtO3JLd1Di0yRmz7g4B0niyY0gzQEd/view?usp=sharing) |

### How to reproduce results
1. Create `checkpoints` directory.
2. Download and extract checkpoint `zip` file under `checkpoints` directory.
3. Evaluate the dataset using the following command.
```
python test.py --config configs/SleePyCo-Transformer_SL-10_numScales-3_{DATASET_NAME}_freezefinetune.json --gpu $GPU_IDs
python test.py --config configs/SleePyCo-Transformer_SL-10_numScales-3_Sleep-EDF-2013_freezefinetune.json --gpu 0
```

## Authors
- **SeongjuLee** [[GoogleScholar](https://scholar.google.com/citations?user=Q0LR04AAAAAJ&hl=ko)] [[GitHub](https://github.com/SeongjuLee)]
- **Yeonguk Yu** [[GoogleScholar](https://scholar.google.com/citations?user=Ctm3p8wAAAAJ&hl=ko)] [[GitHub](https://github.com/birdomi)]
- **Seunghyeok Back** [[GoogleScholar](https://scholar.google.com/citations?user=N9dLZH4AAAAJ&hl=ko)] [[GitHub](https://github.com/SeungBack)]
- **Hogeon Seo** [[GoogleScholar](https://scholar.google.co.kr/citations?user=4llqDpUAAAAJ&hl=ko)] [[GitHub](https://github.com/hogeony)]
- **Kyoobin Lee** (Corresponding Author) [[GoogleScholar](https://scholar.google.com/citations?hl=ko&user=QVihy5MAAAAJ)]

## License

The source code of this repository is released only for academic use. See the [license](./LICENSE) file for details.

## Citation
```
@article{lee2022sleepyco,
  title={SleePyCo: Automatic Sleep Scoring with Feature Pyramid and Contrastive Learning},
  author={Lee, Seongju and Yu, Yeonguk and Back, Seunghyeok and Seo, Hogeon and Lee, Kyoobin},
  journal={arXiv preprint arXiv:2209.09452},
  year={2022}
}
```

## Acknowledgments
This research was supported by a grant from the Institute of Information and Communications Technology Planning and Evaluation (IITP) funded by the Korean government (MSIT) (No. 2020-0-00857, Development of cloud robot intelligence augmentation, sharing and framework technology to integrate and enhance the intelligence of multiple robots). Furthermore, this research was partially supported by the Korea Institute of Energy Technology Evaluation and Planning (KETEP) grant funded by the Korean government (MOTIE) (No. 20202910100030).
