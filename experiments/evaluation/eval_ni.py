import os
import json
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

gemma_template = "{{ bos_token }}{% if messages[0]['role'] == 'system' %}{{ raise_exception('System role not supported') }}{% endif %}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if (message['role'] == 'assistant') %}{% set role = 'model' %}{% else %}{% set role = message['role'] %}{% endif %}{{ '<start_of_turn>' + role + '\n' + message['content'] | trim + '<end_of_turn>\n' }}{% endfor %}{% if add_generation_prompt %}{{'<start_of_turn>model\n'}}{% endif %}"

def build_prompts(args, tokenizer, limit=0):
    with open(args.test_jsonl, "r") as f:
        data = [json.loads(line) for line in f]
    if tokenizer.chat_template is None:
        print(f"chat_template not provided in the tokenizer, using format {args.format}")
        if args.format.lower() != "gemma":
            raise NotImplementedError(f"format [{args.format}] not implemented")
        tokenizer.chat_template = gemma_template
    else:
        print(f"Using chat_template from tokenizer")
    # just take the first row, as it is the user-prompt, and we assume there's only one prompt-response pair
    if limit>0:
        data = data[0:limit]
    raw_prompts = [x['conversations'][0] for x in data]
    ground_truth = [x['conversations'][1]['content'] for x in data]
    prompts = [tokenizer.apply_chat_template([x], tokenize=False, add_generation_prompt=True) for x in tqdm(raw_prompts)]
    return raw_prompts, prompts, ground_truth

def build_llm(args):
    sampling_params = SamplingParams(temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens)
    llm = LLM(model=args.hf_model, revision=args.revision)
    return llm, sampling_params

def evaluate(args):
    print(args)
    print(f"---- build prompts ----")
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model, revision=args.revision)
    raw_prompts, prompts, ground_truth = build_prompts(args, tokenizer)
    print(f"---- build LLMs ----")
    llm, sampling_params = build_llm(args)
    print(f"---- generation starts... ----")
    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
    results = []
    for i in range(len(outputs)):
        result = {
            'prompt': raw_prompts[i],
            'output': outputs[i].outputs[0].text,
            'ground_truth': ground_truth[i]
        }
        results.append(result)
    # make dirs if not exists
    os.makedirs(os.path.dirname(args.output_jsonl), exist_ok=True)
    with open(args.output_jsonl, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
    

    
if __name__=="__main__":
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-model", type=str, default="google/gemma-2b-it")
    parser.add_argument("--revision", type=str, default="main")
    parser.add_argument("--test-jsonl", type=str, default=".cache/datasets/prepared/ni_test.jsonl")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--output-jsonl", type=str, default="outputs.jsonl")
    parser.add_argument("--format", type=str, default="gemma")
    
    evaluate(parser.parse_args())