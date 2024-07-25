# PyTorch-KLD

Kullback-Leibler divergence estimation via variational bounds optimization in PyTorch.

## Implemented estimators

- [x] Donsker-Varadhan
      $$
          \def\proba{\mathbb{P}}
          \DeclareMathOperator{\expect}{\mathbb{E}}
          \DeclareMathOperator{\dispersion}{\mathbb{D}}
          \newcommand{\DKL}[2]{D_{\textnormal{KL}}\left( #1 || #2 \right)}
          \DKL{P}{Q} = \sup_T \left[ \expect_P \, T - \log \expect_Q \exp(T) \right]
      $$
- [x] Nguyen-Wainwright-Jordan
      $$
          \DKL{P}{Q} = \sup_T \left[ \expect_P \, T - \expect_Q \exp(T-1) \right]
      $$
- [x] Nishiyama
      $$
          \DKL{P}{Q} \geq \frac{A - 2 \dispersion_P \, T}{D} \tanh^{-1} \frac{D}{A} + \frac{1}{2} \log \frac{\dispersion_P \, T}{\dispersion_Q \, T},
      $$
      $$
          A = (\expect_Q T - \expect_P T)^2 + \dispersion_Q T + \dispersion_P T, \qquad D = \sqrt{A^2 - 4 \dispersion_Q T \dispersion_P T}
      $$
- [ ] InfoNCE

## Mutual Information Neural Estimation

PyTorch-KLD can be used to estimate the mutual information between random vectors $X$ and $Y$ via estimating $D_\text{KL}(\mathbb{P}_{X,Y}||\mathbb{P}_X \otimes \mathbb{P}_Y)$.
See [`mutual_information.py`](./source/python/torchkld/mutual_information.py)