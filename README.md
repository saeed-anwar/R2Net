# Attention Based Real Image Restoration
This repository is for Attention Based Real Image Restoration introduced in the following paper

[Saeed Anwar](https://saeed-anwar.github.io/),  Nick Barnes, and Lars Petersson, "Attention Based Real Image Restoration", [arXiv, 2020](http://arxiv.org/abs/2004.13524) 

## Contents
1. [Introduction](#introduction)
2. [Requirements](#requirements)
3. [Super-resolution](#super-resolution)
4. [Rain-Removal](#rain-removal)
5. [JPEG-Compression](#jpeg-compression)
6. [Real-Denoising](#real-denoising)
7. [Citation](#citation)
8. [Acknowledgements](#acknowledgements)

## Introduction
Deep convolutional neural networks perform better on images containing spatially invariant degradations, also known as synthetic degradations; however, their performance is limited on real-degraded photographs and requires multiple-stage network modeling. To advance the practicability of restoration algorithms, this paper proposes a novel single-stage blind real image restoration network (R<sup>2</sup>Net) by employing a modular architecture. We use a residual on the residual structure to ease low-frequency information flow and apply feature attention to exploit the channel dependencies. Furthermore, the evaluation in terms of quantitative metrics and visual quality for four restoration tasks, i.e., Denoising, Super-resolution, Raindrop Removal, and JPEG Compression on  11 real degraded datasets against more than 30 state-of-the-art algorithms demonstrate the superiority of our R<sup>2</sup>Net. We also present the comparison on three synthetically generated degraded datasets for denoising to showcase our method's capability on synthetics denoising. 


## Requirements
- The model is built in PyTorch 0.4.0, PyTorch 0.4.1 
- Tested on Ubuntu 14.04/16.04 environment 
- python 3.6
- CUDA 9.0 
- cuDNN 5.1 
- pytorch=0.4.1
- torchvision=0.2.1
- imageio
- pillow

## Super-resolution
The proposed network produces remarkably higher numerical accuracy and better visual image quality than the classical state-of-the-art and CNN algorithms when being evaluated on the three conventional benchmark and three real-world datasets

<p align="center">
  <img width="500" src="https://github.com/saeed-anwar/R2Net/blob/master/Figs/R2Net-SR.png">
</p>

Denoising results: In the first row, an image is corrupted by the Gaussian noise with σ = 50 from the BSD68 dataset. In the second row, a sample image from the RNI15 real noisy dataset. Our results have the best PSNR score for synthetic images, and unlike other methods, it does not have over-smoothing or over-contrasting artifacts. 


### Test
1. Download the trained models and code of our paper from [Google Drive](https://drive.google.com/file/d/1DV9-OgvYoR4ELQZY-R7vZiX5nTf-NZtX/view?usp=sharing). The total size for all models is **3.1MB.**

2. cd to '/IERDTestCode/code', run the following scripts and find the results in directory **IERD_Results**.

    **You can use the following script to test the algorithm. The first script is without self-ensembler and the second one is with self-ensemble.**

``` #Normal
CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --noise_g 1 --model IERD --n_feats 64 --pre_train ../trained_model/IERD.pt --test_only --save_results --save 'SSID_Results' --testpath ../noisy --testset SIDD
```

``` #Ensemble
CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --noise_g 1 --model IERD --n_feats 64 --pre_train ../trained_model/IERD.pt --test_only --save_results --save 'SSIDPlus_Results' --testpath ../noisy --testset SIDD --self_ensemble
```


### Results
**All the results for IERD can be downloaded from GoogleDrive from**  [SSID](https://drive.google.com/file/d/1em70fbrVCggxdv1vi0dLqriAR_f2lPjc/view?usp=sharing) (118MB), [RNI15](https://drive.google.com/file/d/1NUmFpS7Zl4f70OZJVd96t35wSGSyvfMS/view?usp=sharing) (9MB) and [DnD](https://drive.google.com/file/d/1IfTi6ZImNsrzqC6oFhgFF8Z9QKvZeAfE/view?usp=sharing) (2.3GB). 

### DnD Results

Comparison of our method against the state-of-the-art algorithms on real images containing Gaussian noise from Darmstadt Noise Dataset (DND) benchmark for different denoising algorithms. Difference can be better viewed in magnified view.
<p align="center">
  <img width="800" src="https://github.com/saeed-anwar/IERD/blob/master/FIgs/DnDFig.png">
</p>

Mean PSNR and SSIM of the denoising methods evaluated on the real images dataset
<p align="center">
  <img width="400" src="https://github.com/saeed-anwar/IERD/blob/master/FIgs/DnDTable.png">
</p>


## Rain Removal

## Ablation Studies



## Citation
If you find the code helpful in your resarch or work, please cite the following papers.
```
@article{Anwar2020R2NET,
    title={Attention Prior for Real Image Restoration},
    author={Saeed Anwar and Nick Barnes and Lars Petersson},
    year={2020},
    eprint={2004.13524},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
@article{anwar2019ridnet,
  title={Real Image Denoising with Feature Attention},
  author={Anwar, Saeed and Barnes, Nick},
  journal={IEEE International Conference on Computer Vision (ICCV-Oral)},
  year={2019}
}

@article{Anwar2020IERD,
  author = {Anwar, Saeed and Huynh, Cong P. and Porikli, Fatih },
    title = {Identity Enhanced Image Denoising},
    journal={IEEE Computer Vision and Pattern Recognition Workshops (CVPRW)},
    year={2020}
}
```
## Acknowledgements
This code is built on [RIDNET (PyTorch)](https://github.com/saeed-anwar/RIDNet)

