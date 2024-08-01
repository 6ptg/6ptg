# 6PTG

This is the anonymous public code repository of 6PTG.

6PTG is a prefix-level IPv6 target generation algorithm. Utilizing ViT AE and Guided Diffusion model, 6PTG implements target generation based on addressing pattern knowledge under prefix.

## Enviroment requirements

The training and testing experiments are based on Python 3.8.15 and pytorch 1.13.1.

The specific package version can refer to requirements.txt.

We recommend installing pytorch related environments first. Next, install the 0.3.5 version of rotary embedding torch and the 1.10.7 version of vector quantity pytorch. Finally, install other libraries.

Specific versions of pytorch can be found in https://pytorch.org/get-started/previous-versions/.

The following is an example of an installation based on the CPU version.

```shell
pip install torch==1.13.1+cpu torchvision==0.14.1+cpu torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cpu
pip install vector-quantize-pytorch==1.10.7
pip install rotary-embedding-torch==0.3.5
pip install einops==0.8.0 kornia==0.7.3 resize-right==0.0.2 tqdm==4.66.4 transformers==4.43.3
```

## Usage

### Init seed set preparation

Make sure your original address file is stored in the **init_seed_set** folder.

### Expansion scanning

Modify and run **expansion_scanning_generation.py** to generate the expansion scanning addresses.

```shell
python expansion_scanning_generation.py
```

### Final seed set preparation

Store the complete seed address file in the **final_seed_set** folder.

### Model input construction

Modify and run **input_partial.py** and **tensor_construct.py** in turn to build the training input.

```shell
python input_partial.py
python tensor_construct.py
```

### Model training

Then modify and run **suffix_vit-ae_train.py**, **prefix_vit-ae_train.py** and **diffusion_train.py** to train the model. The model is saved every 5 rounds.

```shell
python suffix_vit-ae_train.py
python prefix_vit-ae_train.py
python diffusion_train.py
```

### Prefix target generation

Modify and run **generation.py** to generate. Enter the 64-bit binary network prefix and the model will perform the generation task.

```shell
python generation.py
```

Please perform the specific parameter modification in the py file, which we provide at the beginning of each file main for general modification.

