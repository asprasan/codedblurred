# Video reconstruction by spatio-temporal fusion of blurred-coded image pair
This repository contains inference code for **Video reconstruction by spatio-temporal fusion of blurred-coded image pair** accepted at ICPR 2020.

Preprint version of the paper is available [here](https://arxiv.org/abs/2010.10052)

### Dependencies
+ python v3.6.8
+ pytorch v1.4.0
+ numpy v1.16.4
+ skimage v0.15.0

### Input images and model weights
+ ```data/test_videos```: contains 14 test video sequences of 9 frames each, randomly selected from GoPro dataset and used for evaluation; input to the network is obtained by coded exposure of these frames
+ ```weights```: Download the model weights to this directory from the following paths:
- Weights for coded-blurred input: [download model](https://drive.google.com/file/d/1HsNNWn7SHFR_ubFFjarQFih4HvD-qE7s/view?usp=sharing)
- Weights for single coded input: [download model](https://drive.google.com/file/d/1XgqLUQP1bOkjDme9r1UG3jLCc_B27OZx/view?usp=sharing)
- Copy the downloaded ```.tar.gz``` files to the ```weights``` directory and extract using 
```sh
tar -xvzf single-coded-inp.tar.gz
tar -xzvf coded-blurred-inp-attn.tar.gz
```

### Video reconstruction from coded-blurred image pair
Command to run inference on test videos in ```data/test_videos``` and optionally save results in ```recon_results```:
```sh
python recon_cb.py --savepath recon_results
```
### Video reconstruction from single coded exposure image
Command to run inference on test videos in ```data/test_videos``` and optionally save results in ```recon_results```:
```sh
python recon_sc.py --savepath recon_results
```


### Supplementary material
Video reconstruction results from the paper can be viewed in the supplementary material [here](https://drive.google.com/file/d/1u99_tjrFW56qvVXm46CmA-zBOvI1-56o/view?usp=sharing)

### Bibtex
```
@article{shedligeri2020video,
  title={Video Reconstruction by Spatio-Temporal Fusion of Blurred-Coded Image Pair},
  author={Shedligeri, Prasan and Pal, Abhishek and Mitra, Kaushik and others},
  journal={arXiv preprint arXiv:2010.10052},
  year={2020}
}
```