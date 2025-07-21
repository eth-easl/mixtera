# Full training guide using Mixtera and Torchtitan

This guide will walk you through a full model training using Mixtera on the SlimPajama dataset. We will rely [our fork](https://github.com/eth-easl/torchtitan-mixtera) of the torchtitan training framework, but besides setting up the training configuration, most of this transfers straightforwardly also to other frameworks. 

## 1. Environment setup

We start by setting up dependencies and Mixtera. If you run on bare metal, you can follow the general setup. If you are a SwissAI/CSCS/Alps/Clariden user, we also have prepared specific instructions.

### 1.1 General setup

We first need to install (micro)mamba, if you have not installed it already, to create a Python environment, and clone the Mixtera repository.

```bash
# In case you don't have a Python environment yet
# macos:
brew install micromamba
# alternatively:
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)
# you might also consider pixi (which we use in CI)

# Run this if you have Python already set up
git clone https://github.com/eth-easl/mixtera
cd mixtera

pip install -e ".[torch,dev]"
```

Note that cloning the repo is not necessariy if you do not want to modify  Mixtera's code. In that case, you could also run `pip install git+https://github.com/eth-easl/mixtera[torch,dev]`.

After restarting your shell, you should have access to the `mixtera-server` command. Next, also clone the torchtitan-mixtera repository and install it:

```bash
git clone https://github.com/eth-easl/torchtitan-mixtera
cd torchtitan-mixtera
pip install -e .
```

### 1.2 CSCS alps/clariden

If you are running on clariden--the CSCS Swiss AI cluster--we have prepared a [Dockerfile](clariden/Dockerfile) to build a container with all relevant dependencies. Start by copying that Dockerfile to clariden. Then, [ensure you have set up access to NGC](https://github.com/swiss-ai/documentation/blob/main/pages/setup_ngc.md). Obtain a new node and build the container

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

Note that this script downloads each chunk in a separate directory. For Mixtera, we currently require all files to be stored within a single directory, so make sure to `mv * ..` all files one level upwards in case you use more than one chunk.

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

We can now register the dataset, either on our local server (3.1) or, if you are on clariden (3.2), on a node with Mixtera installed. To this end, copy&paste the following into a file, e.g., `register_slimpj.py`:

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

When running the script, you should see a bunch of `Still waiting for dataset registration to finish at server.` until the registration finishes. In the server logs, you should see a couple of messages like `Processed chunk N` until `All tasks finished.`. Congratulations! You have succesfully registered the samples in the server.

## Starting the training job

We explain briefly how to run torchtitan trainings using Mixtera. We have extended [our fork](https://github.com/eth-easl/torchtitan-mixtera) to support various Mixtera options via the command line. We try to keep the fork synchronized with upstream torchtitan. Currently, pipeline parallelism / context parallelism are not supported since they use different code paths which don't support bfloat16 training. Before actually running the training, we need to define the job in torchtitan's configuration format. For example, you can find the [1.6B model configuration here](https://github.com/eth-easl/torchtitan-mixtera/blob/main/train_configs/ado_1b_default.toml). Important things to adjust include:

- `job.dump_folder` describes where torchtitan will store its model outputs. Ensure this is writable by your training node
- `training.data_parallel_shard_degree` and `training.data_parallel_replicate_degree` will need to be synchronized with the overall number of training nodes that you will train on. For example, if you plan to train across 4 nodes with 4 GPUs each, if you set `data_parallel_replicate_degree` to 2 and `data_parallel_shard_degree` to -1, you will have 2 replication nodes and 8 shards. We will set the overall number of nodes - in case of clariden - in our launcher script. If you use a different environment for multi-node training, you will need to launch it via your cluster manager accordingly.
- `mixtera.vocab_size` needs to match the vocab size of the tokenizer
- `mixtera.chunk_size` describes the number of samples per chunk (see our paper on Mixtera for more information)
- `mixtera.pile` is only relevant if you train on The Pile. Since we use SlimPajama, we need to hardcode our mixture
- Of course, feel free to adjust all parameters to your liking. Note that in torchtitan, you [modify the model architecture in code, not via the config file](https://github.com/eth-easl/torchtitan-mixtera/blob/main/torchtitan/models/llama/__init__.py). 

As stated in the bullet above, in the current version of the code base, we only have hardcoded mixtures for The Pile for our experiments we ran in the paper submission. You will want to [modify the mixture](https://github.com/eth-easl/torchtitan-mixtera/blob/3481b3a95564c2992260b5fa8903eecd94de372b/train.py#L231) we pass to the `QueryExecutionArgs`. You can either define a custom static mixture, or just use an `InferringMixture`. To this end, modify the `train.py` file accordingly to your desired mixture.

What is _very important_ to understand are the mixture processing modes. We describe them the Mixtera paper. Our current implementation enforces the token mixture mode, where Mixtera yields tokenized sequences to the training framework. This is because the torchtitan code was written in a way that the dataset should provide tokenized samples, not samples on the string level. We could extend this in the future. For the implementation of the processing modes, we refer to the [ResultChunk class implementation](https://github.com/eth-easl/mixtera/blob/main/mixtera/core/query/result_chunk.py).

### Starting locally

In the `mixtera` section of the config file, you should add the fields `ip` (default `127.0.0.1`) and `port`(default `8080`) and set them accordingly. You can then kick off a training using `CONFIG_FILE=<path to your toml> ./run_llama_train.sh`.

### Starting on Alps/clariden

We provide a [helper script](https://github.com/eth-easl/mixtera-clariden/blob/main/run_clients.py) for this in our mixtera-clariden repository. Similar to the dataset registration, you first need to edit the `client.yaml` file accordingly. `torchtitan_src` should point to your clone of the torchtitan-mixtera repository, and `mixtera_server_config` should point to the file that configured the server. We use this information to know the IP/node of the server. The `config_file` points to the torchtitan configuration file. Importantly, set `slurm.nodes` such that the number of nodes (with 4 GPUs per node) is in line with your torchtitan configuration. You can then `python run_clients.py client.yaml` to start the training jobs.

### Dynamic mixture using ADO

This section is TBD, but at least in torchtitan, it's basically a matter of providing the initial mixture and selecting ADO as the dynamic mixture. It's all implemented with the per-domain loss.

## Using a different training framework

This section is TBD, but if you take a look at train.py / compare our fork to torchtitan, you should already get a good impression of the changes necessary to support Mixtera.