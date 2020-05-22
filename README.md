# Gretel

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.2597008.svg)](https://doi.org/10.5281/zenodo.2597008)

Implementation of the paper [Extrapolating paths with graph neural networks](http://arxiv.org/abs/1903.07518),

by Jean-Baptiste Cordonnier and Andreas Loukas.

![Hansel und Gretel](images/220px-Hansel-and-gretel-rackham.jpg)

*Illustration by Arthur Rackham, 1909*


## Introduction

We consider the problem of path inference: given a path prefix, i.e., a partially observed sequence of nodes in a graph, we want to predict which nodes are in the missing suffix. In particular, we focus on natural paths occurring as a by-product of the interaction of an agent with a network -- a driver on the transportation network, an information seeker in Wikipedia, or a client in an online shop. Our interest in path inference is due to the realization that, in contrast to shortest-path problems, natural paths are usually not optimal in any graph-theoretic sense, but might still follow predictable patterns.

Our main contribution is a graph neural network called Gretel. Conditioned on a path prefix, this network can efficiently extrapolate path suffixes, evaluate path likelihood, and sample from the future path distribution. Our experiments with GPS traces on a road network and user-navigation paths in Wikipedia confirm that Gretel is able to adapt to graphs with very different properties, while also comparing favorably to previous solutions.

![example on 2D graph](images/planar.png)

## Reproductibility

Install a conda environment and the dependencies with `./create_env.sh env_name`.

Data can be downloaded from [zenodo](https://zenodo.org/record/2597008#.XI9kZS3MzOQ) and is contained in `workspace.zip`.

You can format your own data following the format defined in `main.py:load_data()` documentation.

### Directory structure

```
.
├── gretel
|   ├── config
|        ├── wiki_nll
|        └── ...
└── workspace
|   ├── chkpt
|        ├── trained_model1
|        └── ...
|   ├── mesh
|   ├── planar
|   ├── gps
|   ├── gps-rnn
|   └── wikispeedia
```

Run

```bash
python main.py config/wiki...
```

## Reference

If you find this useful, please consider citing the following:

```
@inproceedings{ijcai2019-303,
   title     = {Extrapolating Paths with Graph Neural Networks},
   author    = {Cordonnier, Jean-Baptiste and Loukas, Andreas},
   booktitle = {Proceedings of the Twenty-Eighth International Joint Conference on Artificial Intelligence, {IJCAI-19}},
   publisher = {International Joint Conferences on Artificial Intelligence Organization},
   pages     = {2187--2194},
   year      = {2019},
   month     = {7},
   doi       = {10.24963/ijcai.2019/303},
   url       = {https://doi.org/10.24963/ijcai.2019/303},
}
```
