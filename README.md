# CGMNet

CGMNet (Coarse-Grained Molecular Network) is a fragment-based pre-training framework for polymer property prediction.

## About

This repository contains the official implementation of CGMNet, including the code and scripts for data preparation, pre-training, fine-tuning, evaluation, and representation export. To facilitate reproducibility, the datasets and data splits used in this work are also organized in this repository. In particular, the `jobs/` directory stores the original data files, together with the processed `.npy` and `.csv` files used in pre-training and downstream experiments. The following sections describe how to use the codebase and provide download links and reproduction guidelines for the datasets, model checkpoints, and processing pipeline.

## Overview of the framework

<p align="center">
  <img src="cgmnet/CGMNet.png" width="800">
</p>

## Environment

Please first set up the runtime environment for CGMNet.

```bash
# TODO
````

## Pre-training data preparation

This section describes how to prepare the data used for pre-training, including vocabulary construction, feature generation, and LMDB building.

### Step 1: Build vocabulary

```bash
# TODO
```

### Step 2: Generate features

```bash
# TODO
```

### Step 3: Build LMDB files

```bash
# TODO
```

## Pre-training

This section describes how to pre-train CGMNet from scratch, or how to use the released pre-trained checkpoints.

```bash
# TODO
```

## Fine-tuning data preparation

This section describes how to prepare downstream datasets and data splits for fine-tuning experiments.

```bash
# TODO
```

## Fine-tuning

This section describes how to fine-tune CGMNet on downstream property prediction tasks.

```bash
# TODO
```

## Extracting molecular and fragment representations

This section describes how to use CGMNet to extract molecular-level and fragment-level representations for customized datasets.

```bash
# TODO
```

## Applications in complex polymer systems

This section describes how to apply CGMNet to complex polymer systems, including the corresponding data organization and usage pipeline.

```bash
# TODO
```

## Reproducibility of the results reported in the paper

This section provides the information required to reproduce the main results reported in our paper, including dataset splits, processed files, checkpoints, and evaluation settings.

### Datasets and processed files

The datasets and data splits used in this work are organized under the `jobs/` directory. This includes the original data files, as well as the processed `.npy` and `.csv` files used in pre-training and downstream experiments.

```bash
# TODO
```

### Model checkpoints

Download links and instructions for the released model checkpoints will be provided here.

```bash
# TODO
```

### Reproducing the reported results

Detailed reproduction commands and experimental settings will be provided here.

```bash
# TODO
```

## Citation

If you find this repository useful, please cite our paper.

```bibtex
# TODO
```

## License
CGMNet is licensed under the Apache License, Version 2.0. See the [LICENSE](LICENSE) file for details.
