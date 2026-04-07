---
title: "FlashQuad: A Python-based multi-backend integration package for machine learning and scientific computing"
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

We developed a numerical integration package based on Python, capable of providingggggg

We developed a numerical integration package for arbitrary dimensions based on PyTorch, capable of providing faster parallel calculation speed and less memory usage compared to torchquad on GPU. By incorporating with PyTorch, this package inherits the automatic differentiation capability, making it ideal for gradient-based optimization methods, such as gradient descent and neural network, i.e. for solving inverse problems and training machine-learning models. Furthermore, this package supports vectorized integration by leveraging broadcasting mechanism of PyTorch, eliminating explicit time-consuming for-loop computations—another improvement over torchquad. Additionally, this package supports integration over complicated boundaries beyond simple hyper-rectangular domains, with tunable sampling points among different dimensions, addressing another two limitations of torchquad. Lastly, with its user-friendly API, this package is easy to adopt, making it a powerful tool for tackling complex integration problems efficiently and driving advancements in scientific computation community.

# Statement of Need
Integration is a fundamental operation in modern numerical scientific research. Most existing software or packages supporting integration primarily run on CPUs, e.g. SciPy in Python, GSL in C/C++, `Integrals.jl` in Julia, `evalf(Int(...))` in Maple, `integral(...)` in MATLAB, and `NIntegrate[...]` in Mathematica. 

With more researchers with computational science background joining machine learning community, there is a growing demand for computations and algorithms running on GPUs—e.g. numerical integration.

On the other hand, numerical integration is essentially a summation problem, which aligns well with the hardware characteristics of GPUs. This means not only faster computation speed, but also more efficient vectorized execution, especially when tackling high-dimensional or large-scale integration tasks.

However, compared with mature CPU-based libraries, the ecosystem of GPU-native numerical integration remains comparatively small. Representative options include `torchquad` based on PyTorch, ... based on TensorFlow, and custom GPU implementations built with CuPy or CUDA. 

These approaches still have notable limitations: they rarely match the algorithmic breadth and mature error-control strategies of established CPU quadrature; many workflows emphasize fixed sampling or Monte Carlo schemes rather than general-purpose adaptive quadrature; support for irregular domains and end-to-end differentiable pipelines can remain cumbersome; and realized speed depends on problem scale, with host–device transfers and GPU memory as frequent bottlenecks.


Here we develop a Python-based integration package for multi-backend , highlighting in . Compared with 

# Comparison and Performance

# References