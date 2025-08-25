# PyTorch-FD

PyTorch $f$-divergence estimation via variational bounds optimization.

## Implemented estimators

- [x] Donsker-Varadhan
- [x] Nguyen-Wainwright-Jordan
- [x] Nishiyama
- [x] InfoNCE

## Mutual Information Neural Estimation

PyTorch-FD can be used to estimate the mutual information between random vectors $X$ and $Y$ via estimating $D_\text{KL}(\mathbb{P}_{X,Y} \Vert \mathbb{P}_X \otimes \mathbb{P}_Y)$.
See [`mutual_information.py`](./source/python/torchfd/mutual_information.py).