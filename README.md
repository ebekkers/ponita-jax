# ✨ 🐴 🔥 PONITA-JAX

This is first go at a JAX implementation of Ponita and is not thoroughly tested, in fact, it is very much hacked into something that seems to reproduce the QM9 results. The repo contains two implementations, one based dense tensors in which (fully connected) graphs are padded to a fixed size and a mask is used to identify the nodes and a standard message passing approach based on a scatter operator. The former can be jitted, the latter cannot—-at least not in it's current form, but perhaps [jraph](https://github.com/google-deepmind/jraph) provides a solution. 

See [the original github repo](https://github.com/ebekkers/ponita) for a [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/) implementation. The original repo has more features than this one. The current repository is a minimal dependency implementation that currently only implements the fiber bundle method. Moreover, the dataloaders do not rely on PyTorch Geometric, but provide the same type of objects (graphs consisting of the tensors: x, pos, edge_index, batch).

## Conda environment
In order to run the code in this repository install the following conda environment
```
conda create --yes --name ponita-jax python=3.11
pip install -U "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
pip3 install wandb
pip3 install hydra-core
pip3 install pandas
pip3 install rdkit
pip3 install tqdm
pip3 install optax
pip3 install flax
```
