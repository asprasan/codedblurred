# VideoReconCodedBlurred
Video reconstruction by spatio-temporal fusion of blurred-coded image pair.

### Dependencies
+ python v3.6.8
+ pytorch v1.4.0
+ numpy v1.16.4
+ skimage v0.15.0

### Video reconstruction from coded-blurred image pair
To run inference on test videos and optionally save results in the specified folder:
```python
python recon_cb.py --savepath results
```
### Video reconstruction from single coded exposure image
To run inference on test videos and optionally save results in the specified folder:
```python
python recon_sc.py --savepath results
```
