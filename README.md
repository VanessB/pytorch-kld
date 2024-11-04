# PyTorch-KLD

Kullback-Leibler divergence estimation via variational bounds optimization in PyTorch.

## Implemented estimators

- [x] Donsker-Varadhan
- [x] Nguyen-Wainwright-Jordan
- [x] Nishiyama
- [x] InfoNCE

## Mutual Information Neural Estimation

PyTorch-KLD can be used to estimate the mutual information between random vectors $X$ and $Y$ via estimating $D_\text{KL}(\mathbb{P}_{X,Y} \Vert \mathbb{P}_X \otimes \mathbb{P}_Y)$.
See [`mutual_information.py`](./source/python/torchkld/mutual_information.py).