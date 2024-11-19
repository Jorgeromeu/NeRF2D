# Scaling Down NeRF with NeRF2D

For more information, see the [Blog Post](https://hackmd.io/@WdAcp83GSrSWsVvpRJWJ2w/SkPKXfsS0)


Code for **NeRF2D**, a 2D analogue of NeRF designed for facilitating experimentation with Neural Radiance Fields, and Novel View Synthesis algorithms.

NeRF2D is trained with 1D views of a 2D scene, and learns to reconstruct a 2D radiance field. Conceptually, this is the same as 3D novel view synthesis, but requires much less compute, and is conceptually easier to understand and visualize:

![Page 1](https://hackmd.io/_uploads/Byy4Bc2S0.png)

![Page 3](https://hackmd.io/_uploads/r1y0I9nSR.png)

_We show that we can reformulate NeRF in 2D by reconstructing a 2D shape from 1D views of it. Fitting a 2D NeRF is very fast and we propose this as a viable toy dataset for performing quick experimentation on NeRF_

# Generating 2D Novel View Synthesis Datasets

To train a 3D NeRF, we need a 2D multi-view dataset. Since these are not readily available we include a blender script for rendering 1D images of an object:


![Peek 2024-06-16 19-01](https://hackmd.io/_uploads/Bk0oq9hBA.gif)

With the addon, we can generate training, validation and testing datasets of any object, with different distribution of camera poses.

![image](https://hackmd.io/_uploads/Hy5vv03BC.png)

Since each training view is just a 2D line, we visualize all of the views together by concatenating each view horizontally and plotting them together. For the example scene above we get the following:

![image](https://hackmd.io/_uploads/H1PEF02H0.png)



# Experiments

We perform experiments on four testing scenes:

![image](https://hackmd.io/_uploads/BybqGA3H0.png)

## 2D NeRF

We fit a 2D NeRF using 50 views of resolution 100 in under a minute. We show the reconstructed testing views after training each scene:

![image](https://hackmd.io/_uploads/SJwzC0nH0.png)

Additionally, since we are working in 2D space, we can visualize the density field by simply uniformly sampling $x,y$ coordinates and querying the density field over space, enabling us to visualize the reconstructed geometries:

![image](https://hackmd.io/_uploads/Sy9u0AnHC.png)


## Positional Encoding

In NeRF, a critical component to their success was the use of **positional encoding**. The spectral bias of neural networks makes it difficult for them to express high-frequency spatial functions. They found that a simple solution is to pass the coordinates through a positional encoding $\gamma$ before feeding it to the MLP:

![image](https://hackmd.io/_uploads/HywmuNiS0.png)

Where $\gamma(p)$ is a series $L$ of alternating sines and cosines of $p$ with exponentially increasing frequencies, where $L$ is a hyperparameter. In NeRF they found this critical, only fitting a blurry version of the scene without it:

![image](https://hackmd.io/_uploads/BkdSPNiHA.png)

We validated this in NeRF2D, with the "Bunny" scene and unsurprisingly found that without positional encoding $(L=1)$ the learned density field is very low frequency, but as we increase $L$ we can fit higher frequency signals, additionally leading to increased PSNR:

![image](https://hackmd.io/_uploads/r1nCKEirA.png)
