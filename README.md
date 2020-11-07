# Video reconstruction by spatio-temporal fusion of blurred-coded image pair
This repository contains inference code for **Video reconstruction by spatio-temporal fusion of blurred-coded image pair** accepted at ICPR 2020.

### Dependencies
+ python v3.6.8
+ pytorch v1.4.0
+ numpy v1.16.4
+ skimage v0.15.0

### Input images and model weights
+ ```data/test_videos```: contains 14 test video sequences of 9 frames each, randomly selected from GoPro dataset and used for evaluation; input to the network is obtained by coded exposure of these frames
+ ```weights```: contains trained weights for video reconstruction from coded-blurred image pair (```coded-blurred-inp-attn.pth```) and from single coded exposure image (```single-coded-inp.pth```)

### Video reconstruction from coded-blurred image pair
Command to run inference on test videos in ```data/test_videos``` and optionally save results in ```recon_results```:
```python
python recon_cb.py --savepath recon_results
```
### Video reconstruction from single coded exposure image
Command to run inference on test videos in ```data/test_videos``` and optionally save results in ```recon_results```:
```python
python recon_sc.py --savepath recon_results
```
