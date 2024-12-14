# PromptKD: Unsupervised Prompt Distillation for Vision-Language Models 

<!-- This is the official PyToch implementation for "PromptKD: Unsupervised Prompt Distillation for Vision-Language Models." (CVPR 2024) -->


> [**PromptKD: Unsupervised Prompt Distillation for Vision-Language Models**]() <br>
> Zheng Li, Xiang Li#, Xinyi Fu, Xin Zhang, Weiqiang Wang, Shuo Chen, Jian Yang#. <br>
> Nankai University, Ant Group, RIKEN <br>
> CVPR 2024 <br>
> [[Paper](https://arxiv.org/abs/2403.02781)] [[Project Page](https://zhengli97.github.io/PromptKD)]

<hr />

## Running

### Preliminary

1. Create the environment and install Dassl.pytorch library. Please follow the instructions detailed in [INSTALL.md](docs/INSTALL.md).

2. (1) Pre-train your own large teacher CLIP model (See below) or (2) use our publicly released pre-trained teacher ViT-L/14 CLIP models. (**Highly Recommended**)   
Our pre-trained teacher models are publicly available at [[Baidu Yun](https://pan.baidu.com/s/1KNJ1mhNKoxdSli4ZldeZUg?pwd=mjf4)] [[TeraBox](https://terabox.com/s/1X4mxJtSaR8W2lrK5bsrCkg)] [[Google Cloud](https://drive.google.com/drive/folders/1OdQ9WauZmYAzVSUTTw7tIKKChyECIS5B?usp=sharing)]   
(Note that due to cloud space limitations, we only provide a limited number of models in Google Cloud. Sorry.)  
After obtaining the teacher model, unzip these files and place the model in the `./teacher_model` folder.   
The accuracy of each teacher model is shown in Tables 10 and 11 in the supplementary material of the paper.  

3. Download the original ViT-B/16 and ViT-L/14 CLIP model weights from the official OpenAI website. Then place these models in the `./clip` folder.  
[[ViT-B/16 CLIP](https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt)] [[ViT-L/14 CLIP](https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt)]

4. Prepare the dataset. Please follow the instructions detailed in [DATASETS.md](docs/DATASETS.md).

### Train Your Teacher Model (Optional)

In our paper, we default use PromptSRC to pre-train our ViT-L/14 CLIP teacher model. We have already provided the config file in `configs/trainers/PromptSRC/vit_l14_c2_ep20_batch8_4+4ctx.yaml`

If you want to train your own teacher model, first you should change `scripts/promptsrc/base2new_train.sh line 11 CFG=vit_b16_c2_ep20_batch4_4+4ctx` to `vit_l14_c2_ep20_batch8_4+4ctx`.
Then follow the instructions listed in `docs/PromptSRC.md` and run the script.

**Important Note:**  
The accuracy of your own teacher model may vary depending on your computing environment. To ensure that your teacher model is adequate for distillation, please refer to Appendix Table 10 to check whether your model achieves appropriate accuracy. 

If your teacher model cannot achieve the corresponding accuracy or cannot be trained due to computational constraints, I highly recommend that you use our publicly available pre-trained models for distillation.

### Running PromptKD 

#### (1) Base-to-Novel Experiments.

1. The base-to-novel experimental settings are provided in the config file at `configs/trainers/PromptKD/vit_b16_c2_ep20_batch8_4+4ctx.yaml`. You can modify the hyper-parameters in this config file according to your needs.

2. Change the dataset path in `scripts/promptkd/base2new_train.sh line 4` to your current path.

3. Run the commands below to train PromptKD on the specified dataset.

For example:
```
# dataset=imagenet, seed=1 
sh scripts/promptkd/base2new_train.sh imagenet 1

# seed=2
sh scripts/promptkd/base2new_train.sh imagenet 2

# seed=3
sh scripts/promptkd/base2new_train.sh imagenet 3

# dataset=caltech101, seed=1
sh scripts/promptkd/base2new_train.sh caltech101 1
```

4. The output results will be automatically saved at `output/base2new/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed_${SEED}`.
