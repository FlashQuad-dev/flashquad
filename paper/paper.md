---
title: "FlashQuad: A Python-based multi-backend integral package for machine learning and scientific computing"
tags:
  - Integral
  - Differentiable programming
  - Machine Learning
authors:
  - name: Ze Ouyang
    affiliation: 1
  - name: Zijian Yi
    affiliation: 2
  - name: Michael Downer
    affiliation: 1
affiliations:
  - name: Department of Physics, The University of Texas at Austin, USA
    index: 1
  - name: Department of Electrical and Computer Engineering, The University of Texas at Austin, USA
    index: 2
date: 2026
bibliography: paper.bib
---

# Summary

We developed a numerical integration package based on Python, capable of providing 

We developed a numerical integration package for arbitrary dimensions based on PyTorch, capable of providing faster parallel calculation speed and less memory usage compared to torchquad on GPU. By incorporating with PyTorch, this package inherits the automatic differentiation capability, making it ideal for gradient-based optimization methods, such as gradient descent and neural network, i.e. for solving inverse problems and training machine-learning models. Furthermore, this package supports vectorized integration by leveraging broadcasting mechanism of PyTorch, eliminating explicit time-consuming for-loop computations—another improvement over torchquad. Additionally, this package supports integration over complicated boundaries beyond simple hyper-rectangular domains, with tunable sampling points among different dimensions, addressing another two limitations of torchquad. Lastly, with its user-friendly API, this package is easy to adopt, making it a powerful tool for tackling complex integration problems efficiently and driving advancements in scientific computation community.

# Statement of Need

...

# References