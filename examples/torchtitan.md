# Full training guide using Mixtera and Torchtitan

This guide will walk you through a full model training using Mixtera on the SlimPajama dataset. We will rely [our fork](https://github.com/eth-easl/torchtitan-mixtera) of the torchtitan training framework, but besides setting up the training configuration, most of this transfers straightforwardly also to other frameworks. 

## 1. Environment setup

We start by setting up dependencies and Mixtera. If you run on bare metal, you can follow the general setup. If you are a SwissAI/CSCS/Alps/Clariden user, we also have prepared specific instructions.

### 1.1 General setup

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

### 1.2 CSCS alps/clariden

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

## 2. Downloading SlimPajama

We provide a [helper script](download_slim_pajama.py) you can use to download the dataset or chunks of it to a local directory or DFS. If you just want to test Mixtera, downloading one chunk is sufficient. For example, you can run `python download_slim_pajama.py --target-dir /path/to/dataset --chunks 0`.


## 3. Starting the Mixtera Server

Every Mixtera Server stores its database and additional temporary files in a directory. First, you need to decide for a path on where the server stores its data.

### 3.1 Bare-metal / general

Let's assume you want to run the server and train on the same node. You probably want to open a tmux pane, a screen, or something similar to start the mixtera server. In this tmux pane, ensure your mamba environment is still activated, and then just lauch the server via `mixtera-server /path/to/server/dir`. You can also set the port and hostname the server listens on, see `mixtera-server --help`. Afterwards, you should see something like

```
mbther@nid005772:/iopsstor/scratch/cscs/mbther/mixtera$ mixtera-server --port 1234 /tmp/mixtera
2025-02-19 13:24:28.147 | INFO     | __main__:main:46 - Starting server, serving from directory /tmp/mixtera
2025-02-19 13:24:28.149 | DEBUG    | mixtera.core.client.mixtera_client:__init__:138 - Initialized current mixture id to -1.
2025-02-19 13:24:28.149 | INFO     | mixtera.core.datacollection.mixtera_data_collection:_init_database:84 - Initializing database.
2025-02-19 13:24:28.164 | INFO     | mixtera.core.datacollection.mixtera_data_collection:_init_database:119 - Database initialized.
2025-02-19 13:24:28.171 | INFO     | mixtera.core.datacollection.mixtera_data_collection:_vacuum:123 - Vacuuming the DuckDB.
2025-02-19 13:24:28.171 | INFO     | mixtera.core.datacollection.mixtera_data_collection:_vacuum:125 - Vacuumd.
2025-02-19 13:24:28.171 | DEBUG    | mixtera.core.query.query_cache:__init__:18 - Initializing QueryCache at /tmp/mixtera/querycache
2025-02-19 13:24:28.172 | INFO     | mixtera.network.server.server:_run_async:379 - Serving MixteraServer on ('0.0.0.0', 1234)
```

You can now proceed with 3.3 to register the SlimPajama dataset.

### 3.2 Clariden/CSCS

On clariden, you typically want to spawn the server as a slurm job. We provide a [helper script](https://github.com/eth-easl/mixtera-clariden/blob/main/run_server.py) for this in our mixtera-clariden repository. First, you want to clone that repository and edit the `server.yaml`. In particular, you want define a `log_dir` where the server logs will be stored, you want to define a time limit for the server, you want to name your environment (`mixtera` if you followed this guide). You want to point the `server_dir` to the path where Mixtera stores its data, and point the `mixtera_dir` to your clone of the repository. You can then run the server:

```
(base) [clariden][mbther@clariden-ln001 mixtera-clariden]$ python run_server.py server.yaml
Job submission output:
Submitted batch job 185481
```

In your specified log file, you should be able to see `*.out` and `*.err` log files which contain the server's log. After a minute or so, they should also confirm that the server is running:
```
2025-02-19 13:59:07.212 | INFO     | __main__:main:46 - Starting server, serving from directory /iopsstor/scratch/cscs/mbther/emptyserver
2025-02-19 13:59:07.216 | DEBUG    | mixtera.core.client.mixtera_client:__init__:138 - Initialized current mixture id to -1.
2025-02-19 13:59:07.217 | INFO     | mixtera.core.datacollection.mixtera_data_collection:_init_database:84 - Initializing database.
2025-02-19 13:59:07.297 | INFO     | mixtera.core.datacollection.mixtera_data_collection:_init_database:119 - Database initialized.
2025-02-19 13:59:07.308 | INFO     | mixtera.core.datacollection.mixtera_data_collection:_vacuum:123 - Vacuuming the DuckDB.
2025-02-19 13:59:07.308 | INFO     | mixtera.core.datacollection.mixtera_data_collection:_vacuum:125 - Vacuumd.
2025-02-19 13:59:07.309 | DEBUG    | mixtera.core.query.query_cache:__init__:18 - Initializing QueryCache at /iopsstor/scratch/cscs/mbther/emptyserver/querycache
2025-02-19 13:59:07.318 | INFO     | mixtera.network.server.server:_run_async:379 - Serving MixteraServer on ('172.28.15.84', 8088)
```

Note down the server IP to be able to register the dataset.

### 3.3 Registering the dataset

We can now register the dataset, either on our local server (3.1) or, if you are on clariden (3.2), on a node with the mixtera environment activated (and mixtera installed via `pip install -e .`). To this end, copy&paste the following into a file, e.g., `register_slimpj.py`:

```python
from mixtera.core.client import MixteraClient
from mixtera.core.datacollection.datasets import JSONLDataset

def parsing_func(sample):
    import json
    return json.loads(sample)["text"]

if __name__ == "__main__":
    client = MixteraClient.from_remote("172.28.15.84", 8088)

    client.register_dataset(
        "slimpajama", "/capstor/store/cscs/swissai/a09/mixtera/data/slimpajama", JSONLDataset, parsing_func, "SLIM_PAJAMA"
    )
```

Of course, you have to replace the ip address and port of the server, as well as the path of the jsonL files. If you are running local, your IP can be set to `127.0.0.1`, and on clariden, you can find the IP in the server log as mentioned before or you use `squeue` to get the name of the node your server is running on.

If you followed the [examples](client_server_example.py) before, this script probably seems familiar. This script informs the server about where the data lies, and uses the `SLIM_PAJAMA` metadata parser, which is shipped with Mixtera, to parse the metadata information. If you want to use a custom metadata parser, check out the [examples](client_server_example.py), they can easily be defined and registered.


## Starting the training job

It now is time to start a training job,

### Starting locally

### Starting on Alps/clariden

### Dynamic mixture using ADO


## Using a different training framework