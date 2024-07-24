import functools
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import Dataset, Features, IterableDataset, Sequence, Value
from datasets.distributed import split_dataset_by_node
from integrationtests.utils import TestMetadataParser, setup_test_dataset
from mixtera.core.client import MixteraClient
from mixtera.core.client.mixtera_client import QueryExecutionArgs, ResultStreamingArgs
from mixtera.core.datacollection.datasets import JSONLDataset
from mixtera.core.query import ArbitraryMixture, Query
from mixtera.hf import MixteraHFDataset
from transformers import AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "True"


def sample_parsing_func(sample):
    import json

    return json.loads(sample)["text"]


def compare_tensors(tensor1, tensor2):
    if not torch.equal(tensor1, tensor2):
        print(f"Tensors differ: {tensor1} != {tensor2}")
        return False
    return True


def compare_dicts_with_tensor_values(dict1, dict2):
    if dict1.keys() != dict2.keys():
        print(f"Dictionary keys differ: {dict1.keys()} != {dict2.keys()}")
        return False
    if "input_ids" in dict1 and "input_ids" in dict2:
        return compare_tensors(dict1["input_ids"], dict2["input_ids"])
    print("Missing 'input_ids' in one of the dictionaries")
    return False


def compare_lists_of_dicts(list1, list2):
    if len(list1) != len(list2):
        print(f"List lengths differ: {len(list1)} != {len(list2)}")
        return False
    for index, (dict1, dict2) in enumerate(zip(list1, list2)):
        if not compare_dicts_with_tensor_values(dict1, dict2):
            print(f"Difference found at index {index} in the lists.")
            return False
    return True


def group_texts(sequence_length: int, examples: dict[str, list[np.ndarray]]) -> dict[str, list[np.ndarray]]:
    # Taken from https://github.com/MaxiBoether/nanotron-streaming
    concatenated_examples = {k: np.concatenate(v) for k, v in examples.items()}
    total_length = len(concatenated_examples[next(iter(examples.keys()))])
    if total_length >= sequence_length + 1:
        total_length = ((total_length - 1) // sequence_length) * sequence_length + 1

    result = {
        k: [t[i : i + sequence_length + 1] for i in range(0, total_length - (sequence_length + 1), sequence_length)]
        for k, t in concatenated_examples.items()
    }

    return result


def _tokenize_and_group_texts(tokenizer: Any, sequence_length: int, texts: list[str]) -> dict[str, list[np.ndarray]]:
    tokenized_batch = tokenizer.batch_encode_plus(texts, return_attention_mask=False, return_token_type_ids=False)
    tokenized_batch = {k: [np.array(tokenized_texts) for tokenized_texts in v] for k, v in tokenized_batch.items()}
    return group_texts(sequence_length, tokenized_batch)


def clm_process(
    raw_dataset: "Dataset",
    tokenizer: Any,
    text_column_name: str,
    dataset_processing_num_proc_per_process: int,
    dataset_overwrite_cache: bool,
    sequence_length: int,
):

    # some args not supported by the IterableDataset
    additional_args = (
        {
            "num_proc": dataset_processing_num_proc_per_process,
            "load_from_cache_file": not dataset_overwrite_cache,
            "desc": f"Grouping texts in chunks of {sequence_length+1}",
        }
        if not isinstance(raw_dataset, IterableDataset)
        else {}
    )

    if isinstance(raw_dataset, IterableDataset):
        raw_dataset = raw_dataset._resolve_features()

    train_dataset = raw_dataset.map(
        functools.partial(_tokenize_and_group_texts, tokenizer, sequence_length),
        input_columns=text_column_name,
        remove_columns=raw_dataset.column_names,
        features=Features({"input_ids": Sequence(feature=Value(dtype="int64"), length=sequence_length + 1)}),
        batched=True,
        **additional_args,
    )

    return train_dataset


def instantiate_hf_dataloader(
    client: MixteraClient,
    query: Query,
    query_execution_args: QueryExecutionArgs,
    streaming_args: ResultStreamingArgs,
    batch_size: int,
):
    raw_dataset = MixteraHFDataset(client, query, query_execution_args, streaming_args)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    train_dataset = clm_process(
        raw_dataset=raw_dataset,
        tokenizer=tokenizer,
        text_column_name="text",  # by MixteraHFDataset implementation
        dataset_processing_num_proc_per_process=-1,  # will be ignored
        dataset_overwrite_cache=False,  # will be ignored
        sequence_length=10,
    )
    train_dataset = train_dataset.with_format(type="torch")
    train_dataset = split_dataset_by_node(train_dataset, world_size=1, rank=0)
    dl = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=query_execution_args.num_workers)

    return dl


def run_query(
    client: MixteraClient,
    query_exec_args: QueryExecutionArgs,
    batch_size: int,
    tunnel: bool,
):
    job_id = (
        f"hf0_{query_exec_args.mixture.chunk_size}_{batch_size}_{query_exec_args.dp_groups}"
        + f"_{query_exec_args.nodes_per_group}_{query_exec_args.num_workers}_{tunnel}"
    )
    query = Query.for_job(job_id).select(("language", "==", "JavaScript"))
    dl = instantiate_hf_dataloader(
        client, query, query_exec_args, ResultStreamingArgs(job_id=job_id, tunnel_via_server=tunnel), batch_size
    )
    return list(dl)


def test_hfds(server_dir: Path) -> None:
    server_file = setup_test_dataset(server_dir)
    server_client = MixteraClient("127.0.0.1", 6666)

    assert server_client.register_metadata_parser("TEST_PARSER_HF", TestMetadataParser)
    assert server_client.register_dataset(
        "hf_integrationtest_dataset", server_file, JSONLDataset, sample_parsing_func, "TEST_PARSER_HF"
    )
    assert server_client.check_dataset_exists("hf_integrationtest_dataset"), "Dataset does not exist!"

    for batch_size in [1, 500]:
        for num_workers in [0, 8]:
            # The output varies between batch sizes and num workers
            first_exec = True
            prev_data = []
            for mixture in [ArbitraryMixture(x) for x in [1, 2000]]:
                for tunnel in [False]:
                    try:
                        query_exec_args = QueryExecutionArgs(
                            mixture=mixture,
                            num_workers=num_workers,
                        )
                        curr_data = run_query(server_client, query_exec_args, batch_size, tunnel)
                        assert len(curr_data) > 0  # we need some answer
                        print("Query done, start comparing data.")
                        if first_exec:
                            prev_data = curr_data
                            first_exec = False

                        assert compare_lists_of_dicts(prev_data, curr_data), f"{prev_data}\n\n{curr_data}"
                        prev_data = curr_data
                    except Exception as e:
                        print(
                            "Error with "
                            + f"chunk_size = {mixture.chunk_size}, num_workers = {num_workers},"
                            + f"batch_size = {batch_size}, tunnel = {tunnel}"
                        )
                        raise e

    print("Finished huggingface integration test.")


def main() -> None:
    server_dir = Path(sys.argv[1])
    test_hfds(server_dir)


if __name__ == "__main__":
    main()
