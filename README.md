# Hyperbolic Embedding

A repository to investigate how to use Hyperbolic Embeddings in deep learning, with a focus on music generation.

## Summary

Euclidean space has zero curvature. Hyperbolic space has constant negative curvature and naturally captures hierarchical structure, making it well-suited for tree-like and hierarchical data in deep learning.

This repository contains two independent projects:

| Folder | Framework | Description |
|---|---|---|
| [`torchExperiments/`](./torchExperiments/) | PyTorch | Hyperbolic NN experiments on classification and regression tasks |
| [`HyperbolicTimbrenet/`](./HyperbolicTimbrenet/) | TensorFlow | TimbreNet model extended with hyperbolic layers for music generation |

## References

- [Hyperbolic Neural Networks](https://arxiv.org/abs/1805.09112)
- [Hyperbolic Neural Networks ++](https://arxiv.org/pdf/2006.08210.pdf)

## TODO

- [ ] Adapt wrapped hyperbolic layers to the TimbreNet model
- [ ] Train the model with hyperbolic layers
- [ ] Incorporate [Hyperbolic CNN](https://github.com/kschwethelm/HyperbolicCV/tree/main) and [Hyperbolic VAE](https://github.com/julian-8897/hyperbolic_vae/tree/master)

---

Advisors: Denis Parra, Mircea Petrache, Rodrigo Cadiz
