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
- PyTorch 0.4.0, PyTorch 0.4.1 
- Tested on Ubuntu 14.04/16.04 environment 
- torchvision=0.2.1
- python 3.6
- CUDA 9.0 
- cuDNN 5.1 
- imageio
- pillow

---

## Super-resolution
The architecture for super-resolution.

<p align="center">
  <img width="700" src="https://github.com/saeed-anwar/R2Net/blob/master/Figs/R2Net-SR.png">
  <img width="700" src="https://github.com/saeed-anwar/R2Net/blob/master/Figs/EAM.png">

</p>

### SR Test
1. Download the trained models and code of our paper from [Google Drive](). The total size for all models is **??MB.**

2. cd to '/TestCode/code', run the following scripts and find the results in directory **SR_Results**.

    **You can use the following script to test the algorithm.**

``` #Normal
CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --noise_g 1 --model IERD --n_feats 64 --pre_train ../trained_model/IERD.pt --test_only --save_results --save 'SSID_Results' --testpath ../noisy --testset SIDD
```

### SR Results
**All the results for SuperResolution R<sup>2</sup>Net can be downloaded from**  [SET5]() (??MB), [SET12]() (??MB), [BSD100]() (??MB) and [Urban100]() (??GB). 

#### Visual Results

The visual comparisons for 4x super-resolution against several state-of-the-art algorithms on an image from Urban100 dataset. Our R<sup>2</sup>Ne results are the most accurate.
<p align="center">
  <img width="800" src="https://github.com/saeed-anwar/R2Net/blob/master/Figs/R2Net-SR-visual.png">
</p>

#### Quantitative Results

Mean PSNR and SSIM of the denoising methods evaluated on the real images dataset
<p align="center">
  <img width="900" src="https://github.com/saeed-anwar/R2Net/blob/master/Figs/R2Net-SR-psnr.png">
</p>
The performance of super-resolution algorithms on Set5, Set14, BSD100, and URBAN100 datasets for upscaling factors of 2, 3, and 4.
The bold highlighted results are the best on single image super-resolution.

---

## Rain Removal
The architecture for Rain Removal and the subsequent restoration tasks. There are two modifications: the change in position of long skip connection and removal of upsampling
layer.

<p align="center">
  <img width="700" src="https://github.com/saeed-anwar/R2Net/blob/master/Figs/R2Net-all.png">
</p>

### RainRemoval Test
1. The trained models and code for rain removal can be downloaded from [here](https://drive.google.com/file/d/1mlQgVUA1GDTfLjYDPAf13w1YMBGrKMo5/view?usp=sharing). The total size for all models is **121.5MB.**

2. cd to '/R2NetRainRemovalTestCode/code',  either run **bash TestScripts.sh** or run the following individual commands and find the results in directory **R2NET_DeRainResults**.

    **You can use the following script to test the algorithm.**

``` #Normal
# test_a
CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --noise_g 1 --model R2NET --n_feats 64 --pre_train ../trained_models/R2Net_RainRemoval.pt --test_only --save_results --save 'R2NET_test_a' --testpath ../rainy --testset test_a

# test_b
CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --noise_g 1 --model R2NET --n_feats 64 --pre_train ../trained_models/R2Net_RainRemoval.pt --test_only --save_results --save 'R2NET_test_b' --testpath ../rainy --testset test_b
```

### RainRemoval Results
**All the results for  Rain Removal R<sup>2</sup>Net can be downloaded from [here](https://drive.google.com/file/d/1GUJ-2G8rBjeXGaUctHDc6OkkpYmC4T3b/view?usp=sharing) for both DeRain's test_a and test_b datasets.** 

#### Visual Results

The visual comparisons on rainy images. The first figure is showing the plate which is affected by raindrops. Our method is consistent in restoring raindrop affected areas. Similarly, in the second example of a rainy image, the cropped region is showing the road sign affected by raindrops. Our method recovers the distorted colors closer to the ground-truth.
<p align="center">
  <img width="600" src="https://github.com/saeed-anwar/R2Net/blob/master/Figs/R2Net-rain-visual1.png">
  <img width="600" src="https://github.com/saeed-anwar/R2Net/blob/master/Figs/R2Net-rain-visual2.png">
</p>

#### Quantitative Results

<p align="center">
  <img width="500" src="https://github.com/saeed-anwar/R2Net/blob/master/Figs/R2Net-rain-psnr.png">
</p>
The average PSNR(dB)/SSIM from different methods on raindrop dataset.

---

## JPEG Compression
The architecture is same for the rest of restoration tasks.

### JPEG Compression Test
1. Download the trained models and code for JPEG Compression of R<sup>2</sup>Net from [Google Drive](https://drive.google.com/file/d/1sABy-hp60fmJdlk2HxUnat65dxftzR4a/view?usp=sharing). The total size for all models is 43MB.**

2. cd to '/R2NetJPEGTestCode/code', either run **bash TestScripts.sh** or run the following individual commands and find the results in directory **R2Net_Results**.

    **You can use the following script to test the algorithm.**

``` 
# Q10
CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --noise_g 10 --model R2NET --n_feats 64 --pre_train ../trained_models/R2Net_Q10.pt --test_only --save_results --save 'R2NET_JPEGQ10' --testpath ../noisy --testset LIVE1

# Q20
CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --noise_g 20 --model R2NET --n_feats 64 --pre_train ../trained_models/R2Net_Q20.pt --test_only --save_results --save 'R2NET_JPEGQ20' --testpath ../noisy --testset LIVE1

# Q30
CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --noise_g 30 --model R2NET --n_feats 64 --pre_train ../trained_models/R2Net_Q30.pt --test_only --save_results --save 'R2NET_JPEGQ30' --testpath ../noisy --testset LIVE1

# Q40
CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --noise_g 40 --model R2NET --n_feats 64 --pre_train ../trained_models/R2Net_Q40.pt --test_only --save_results --save 'R2NET_JPEGQ40' --testpath ../noisy --testset LIVE1
```

### JPEG Compression Results
**If you don't want to re-run the models and save some computation, then all the results for JPEG Compression R<sup>2</sup>Net can be downloaded from**  [LIVE1](https://drive.google.com/file/d/1TGJgkJoJn6Jhbf0km3YRb_ITos0S3EeU/view?usp=sharing) (51.5MB). 

####  Visual Results

sample images of Monarch and parrot with the artifacts having a quality factor of 20. Our R<sup>2</sup>Net restore texture correctly, specifically the line, as shown in the zoomed version of the restored patch in Monarch images. Moreover, R<sup>2</sup>Net restores the texture accurately on the face of the parrot in the second image.
<p align="center">
  <img width="600" src="https://github.com/saeed-anwar/R2Net/blob/master/Figs/R2Net-jpeg-visual1.png">
   <img width="600" src="https://github.com/saeed-anwar/R2Net/blob/master/Figs/R2Net-jpeg-visual2.png">
</p>

#### Quantitative Results

Average PSNR/SSIM for JPEG image deblocking for quality factors of 10, 20, 30, and 40 on LIVE1 dataset. The best results are in bold.
<p align="center">
  <img width="900" src="https://github.com/saeed-anwar/R2Net/blob/master/Figs/R2Net-jpeg-psnr.png">
</p>

---

## Real Denoising

**The real image denoising can be found [here](https://github.com/saeed-anwar/RIDNet)**

---

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

