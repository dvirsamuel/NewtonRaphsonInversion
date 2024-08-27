# RNRI: Regularized Newton Raphson Inversion for Text-to-Image Diffusion Model

> Dvir Samuel, Barak Meiri, Nir Darshan, Gal Chechik, Shai Avidan, Rami Ben-Ari
> OriginAI, Tel Aviv University, Bar Ilan University, NVIDIA Research

>
>
> Diffusion inversion is the problem of taking an image and a text prompt that describes it, and finding a noise latent that would generate the image. Most current inversion techniques operate by approximately solving an implicit equation, and may converge slowly or yield poor reconstructed images.
Here, we formulate the problem as finding the roots of an implicit equation and design a method to solve it efficiently. Our solution is based on Newton-Raphson (NR), a well-known technique in numerical analysis. A naive application of NR may be computationally infeasible and tends to converge to incorrect solutions. We describe an efficient regularized formulation that converges quickly to solution that provide high-quality reconstructions. We also identify a source of inconsistency stemming from prompt conditioning during the inversion process, which significantly degrades the inversion quality. To address this, we introduce a prompt-aware adjustment of the encoding, effectively correcting this issue.
Our solution, Regularized Newton-Raphson Inversion, inverts an image within 0.5 sec for latent consistency models, opening the door for interactive image editing. We further demonstrate improved results in image interpolation and generation of rare objects.


<a href="https://arxiv.org/abs/2312.12540"><img src="https://img.shields.io/badge/arXiv-2304.14530-b31b1b.svg" height=22.5></a>

<p align="center">
<img src="teaser.gif" width="600px"/>  
<br>

## Requirements

Quick installation using pip:
```
pip install -r requirements.txt
```

## Usage

To run a fast Newton-Raphson inversion (using SDXL-turbo), you can simply run 'main.py':

```
python main.py
```

**Baseline: Fixed-point Inversion.**  
We have also implemented a fast fixed-point inversion for StableDiffuion2.1 (See more details in [our previous paper](https://arxiv.org/pdf/2312.12540v1)).

```
cd src/FPI
PYTHONPATH='.' python main.py
```

## Cite Our Paper
If you find our paper and repo useful, please cite:
```
@misc{samuel2023regularized,
  author    = {Dvir Samuel and Barak Meiri and Nir Darshan and Shai Avidan and Gal Chechik and Rami Ben-Ari},
  title     = {Regularized Newton Raphson Inversion for Text-to-Image Diffusion Models},
  year      = {2024}
}
```
