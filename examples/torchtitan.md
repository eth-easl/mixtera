# Full training guide using Mixtera and Torchtitan

This guide will walk you through a full model training using Mixtera on the SlimPajama dataset. We will rely [our fork](https://github.com/eth-easl/torchtitan-mixtera) of the torchtitan training framework, but besides setting up the training configuration, most of this transfers straightforwardly also to other frameworks. 

## Environment setup

We start by setting up dependencies and Mixtera. If you run on bare metal, you can follow the general setup. If you are a SwissAI/CSCS/Alps/Clariden user, we also have prepared specific instructions.

### General setup

We first need to install (micro)mamba, if you have not installed it already, and clone the Mixtera repository.

```bash
# In case you don't have micromamba yet
# macos:
brew install micromamba
# alternatively:
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)

# Start here if you have micromamba already
git clone https://github.com/eth-easl/mixtera
cd mixtera

micromamba env create -f ./environment.yml
micromamba activate mixtera
pip install -e .
pip install -r dev-requirements.txt
```

At latest after restarting your shell, you should have access to the `mixtera-server` command. Next, also clone the torchtitan-mixtera repository and install it:

```bash
git clone https://github.com/eth-easl/torchtitan-mixtera
cd torchtitan-mixtera
pip install -e .
```

### CSCS alps/clariden

If you are running on clariden, we have prepared a [Dockerfile](clariden/Dockerfile) to build a container with all relevant dependencies. Start by copying that Dockerfile to clariden. Then, [ensure you have set up access to NGC](https://github.com/swiss-ai/documentation/blob/main/pages/setup_ngc.md). Obtain a new node and build the container

```bash
srun --container-writable -p normal --pty bash
cd <the directory with the Dockerfile>
podman build -t mixtera .
enroot import -o $SCRATCH/mixtera.sqsh podman://mixtera
```

After building the sqsh file, you can exit the container and copy your image, e.g., to capstor or iopstor. Now, we need to write an environment definition. To this end, create a file `~/.edf/mixtera.toml` and enter something along those lines, adjusted for your paths, and horizontal:

```
image = "/iopsstor/scratch/cscs/mbther/mixtera_25.sqsh"

mounts = [
  "/users/mbther:/users/mbther",
  "/iopsstor/scratch/cscs/mbther:/iopsstor/scratch/cscs/mbther",
  "/capstor/store/cscs/swissai/a09:/capstor/store/cscs/swissai/a09"
]

workdir = "/users/mbther"

[annotations]
com.hooks.aws_ofi_nccl.enabled = "true"
com.hooks.aws_ofi_nccl.variant = "cuda12"
```

You should be able run `srun --container-writable --environment=mixtera -p normal --pty bash` now and obtain a node with the Mixtera environment set up. Congratulations! We recommend storing the mixtera and torchtitan-mixtera repositories somewhere on your iopstor scratch directories, such that you have access to them from any node.

## Obtaining and registering data

### Downloading SlimPajama

### Registering data in Mixtera

## Starting the training job

### Starting locally

### Starting on Alps/clariden

### Dynamic mixture using ADO