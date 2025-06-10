# ReFactorGNN

Official demo code for implementing the ReFactor GNN. We suggest you start with `refactorgnn_demo_sgd_noreg.py`, the very basic version of ReFactorGNN. Then go to `refactorgnn_demo_sgd.py`, which adds the N3 regularizer. Finally, check out the one induced by AdaGrad rather than SGD `refactorgnn_demo_adagrad.py`, which empirically perform better for muliti-relational link prediction tasks. Refer to our [paper](https://arxiv.org/abs/2207.09980) for more details.


The full code for the experiments can be found in this [repo](https://github.com/yihong-chen/scalable-gpt-gnn/). Use with caution as it is highly customized. Create an issue if you need help running the code. At the same time, a simplified implementation in the future I will possibly try is to use transformer [active forgetting](https://arxiv.org/abs/2307.01163), where the embedding layer is seen as historical cache of states and cleaning them become as simple as cleaning the token/node embeddings. This will potentially avoid the sophisticated message, update, aggregate function in the implementation here. If you are interested in trying this out, send me an email!

If you find the code useful, please cite us by
```
@inproceedings{chen2022refactor,
title={ReFactor {GNN}s: Revisiting Factorisation-based Models from a Message-Passing Perspective},
author={Yihong Chen and Pushkar Mishra and Luca Franceschi and Pasquale Minervini and Pontus Stenetorp and Sebastian Riedel},
booktitle={Advances in Neural Information Processing Systems},
editor={Alice H. Oh and Alekh Agarwal and Danielle Belgrave and Kyunghyun Cho},
year={2022},
url={https://openreview.net/forum?id=81LQV4k7a7X}
}
```
