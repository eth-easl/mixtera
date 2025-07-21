import functools
import multiprocessing as mp
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from datasets import Dataset, Features, IterableDataset, Sequence, Value
from datasets.distributed import split_dataset_by_node
from mixtera_integrationtests.utils import TestMetadataParser, setup_test_dataset
from transformers import AutoTokenizer

import torch
from mixtera.core.client import MixteraClient
from mixtera.core.client.mixtera_client import QueryExecutionArgs, ResultStreamingArgs
from mixtera.core.datacollection.datasets import JSONLDataset
from mixtera.core.query import Query
from mixtera.core.query.mixture import ArbitraryMixture
from mixtera.hf import MixteraHFDataset

# We test both 0 workers and 8 workers
# Instantiating a tokenizer for 0 workers and then again for 8 workers apparently breaks
# Also see https://stackoverflow.com/questions/62691279/how-to-disable-tokenizers-parallelism-true-false-warning/
# and https://github.com/huggingface/transformers/issues/5486.
# For this test this is ok, and in an actual training it's not an issue, because we never use tokenizers before iterating.
os.environ["TOKENIZERS_PARALLELISM"] = "False"


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
    sequence_length: int,
):
    raw_dataset = MixteraHFDataset(client, query, query_execution_args, streaming_args)

    # If using the 'token' mixture type, the data is already tokenized
    if streaming_args.chunk_reading_mixture_type != "token":
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        if mp.get_start_method() == "spawn":
            train_dataset = clm_process(
                raw_dataset=raw_dataset,
                tokenizer=tokenizer,
                text_column_name="text",  # By MixteraHFDataset implementation
                dataset_processing_num_proc_per_process=-1,  # Will be ignored
                dataset_overwrite_cache=False,  # Will be ignored
                sequence_length=sequence_length,
            )
        else:
            print("In a forking environment, using fork for multiprocessing hang when using map! Skipping map.")
            train_dataset = raw_dataset
    if "train_dataset" not in locals():
        train_dataset = raw_dataset

    train_dataset = train_dataset.with_format(type="torch")
    train_dataset = split_dataset_by_node(train_dataset, world_size=1, rank=0)
    # If we tokenize we give it more time because we might need to download the tokenizer.
    timeout_factor = 10 if streaming_args.chunk_reading_mixture_type == "token" else 1
    dl = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=query_execution_args.num_workers,
        timeout=5 * timeout_factor if query_execution_args.num_workers > 0 else 0,
    )

    return dl


def run_query(
    client: MixteraClient,
    query_exec_args: QueryExecutionArgs,
    batch_size: int,
    streaming_args: ResultStreamingArgs,
    sequence_length: int,
):
    job_id = streaming_args.job_id
    query = Query.for_job(job_id).select(("language", "==", "JavaScript"))
    dl = instantiate_hf_dataloader(client, query, query_exec_args, streaming_args, batch_size, sequence_length)
    return list(dl)


def test_hfds(server_dir: Path) -> None:
    setup_test_dataset(server_dir)
    server_client = MixteraClient("127.0.0.1", 6666)

    assert server_client.register_metadata_parser("TEST_PARSER_HF", TestMetadataParser)
    assert server_client.register_dataset(
        "hf_integrationtest_dataset", server_dir, JSONLDataset, sample_parsing_func, "TEST_PARSER_HF"
    )
    assert server_client.check_dataset_exists("hf_integrationtest_dataset"), "Dataset does not exist!"

    # Parameters to test
    batch_sizes = [1, 500]
    num_workers_list = [0, 8]
    mixture_sizes = [1, 2000]
    tunnels = [False]
    mixture_types = ["simple", "token"]
    sequence_length = 10

    for batch_size in batch_sizes:
        for num_workers in num_workers_list:
            for mixture_size in mixture_sizes:
                mixture = ArbitraryMixture(mixture_size)
                for tunnel in tunnels:
                    for mixture_type in mixture_types:
                        for t_en in [False, True]:
                            if mixture_type != "token" and not t_en:
                                continue

                            if mixture_type == "token" and mixture_size == 1:
                                continue

                            try:
                                query_exec_args = QueryExecutionArgs(
                                    mixture=mixture,
                                    num_workers=num_workers,
                                )
                                job_id = (
                                    f"hf_{mixture_type}_{t_en}_{query_exec_args.mixture.chunk_size}_{batch_size}_{query_exec_args.dp_groups}"
                                    + f"_{query_exec_args.nodes_per_group}_{query_exec_args.num_workers}_{tunnel}"
                                )
                                # Set the streaming_args with the current mixture_type
                                streaming_args = ResultStreamingArgs(
                                    job_id=job_id,
                                    tunnel_via_server=tunnel,
                                    chunk_reading_mixture_type=mixture_type,
                                    chunk_reading_sequence_len=sequence_length,
                                    chunk_reading_tokenizer="gpt2",
                                    chunk_reading_tokenization_bs=32,
                                    chunk_reading_token_separate_thread=t_en,
                                    chunk_reading_token_at_least_one_sample=True,
                                )
                                curr_data = run_query(
                                    server_client, query_exec_args, batch_size, streaming_args, sequence_length
                                )
                                assert len(curr_data) > 0, "No data returned from query."

                                # Additional checks for 'token' mixture type
                                if mixture_type == "token":
                                    # Now, check that each batch has the correct sequence length
                                    for batch in curr_data:
                                        input_ids = batch["input_ids"]
                                        # input_ids should be of shape (batch_size, sequence_length+1)
                                        expected_shape = (batch_size, sequence_length + 1)
                                        actual_shape = input_ids.shape
                                        # Handle the case where the last batch might be smaller
                                        if actual_shape[0] != batch_size:
                                            expected_shape = (actual_shape[0], sequence_length + 1)
                                        assert (
                                            actual_shape == expected_shape
                                        ), f"Expected input_ids shape {expected_shape}, got {actual_shape}"

                            except Exception as e:
                                print(
                                    f"Error with mixture_type={mixture_type}, chunk_size={mixture.chunk_size}, "
                                    f"num_workers={num_workers}, batch_size={batch_size}, tunnel={tunnel}, "
                                    f"sequence_length={sequence_length}"
                                )
                                raise e

    print("Finished huggingface integration test with merged loops.")


def main() -> None:
    server_dir = Path(sys.argv[1])
    test_hfds(server_dir)


if __name__ == "__main__":
    main()
