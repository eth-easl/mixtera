
import os
import json
import numpy as np

def select(args):
    print(args)
    with open(args.train_jsonl, "r") as f:
        data = [json.loads(line) for line in f]
    dataset_size = len(data)
    desired_size = int(dataset_size * args.portion)
    categories = set([x['meta']['categories'][0] for x in data])
    count = {x: len([y for y in data if y['meta']['categories'][0] == x]) for x in categories}
    print(f"desired_size: {desired_size}; count: {count}")
    # a human-designed list of categories to select
    """
    desired_size: 240489; count: {'Pos Tagging': 50944, 'Sentence Composition': 9112, 'Preposition Prediction': 926, 'Dialogue Generation': 41839, 'Text Quality Evaluation': 20232, 'Grammar Error Detection': 11530, 'Question Generation': 145603, 'Wrong Candidate Generation': 51636, 'Question Decomposition': 9521, 'Text Categorization': 91572, 'Discourse Relation Classification': 1000, 'Story Composition': 45866, 'Poem Generation': 6442, 'Question Understanding': 59954, 'Coherence Classification': 30077, 'Named Entity Recognition': 10935, 'Negotiation Strategy Detection': 7080, 'Fact Verification': 6553, 'Mathematics': 23817, 'Spelling Error Detection': 6499, 'Intent Identification': 15816, 'Word Relation Classification': 8872, 'Word Semantics': 19294, 'Speaker Identification': 19800, 'Sentiment Analysis': 172791, 'Number Conversion': 998, 'Fill in The Blank': 26786, 'Sentence Ordering': 7184, 'Sentence Perturbation': 9660, 'Summarization': 40437, 'Paraphrasing': 11939, 'Irony Detection': 2854, 'Code to Text': 21328, 'Question Answering': 346685, 'Commonsense Classification': 130257, 'Text Matching': 49297, 'Explanation': 21079, 'Text Simplification': 12619, 'Translation': 1199, 'Dialogue State Tracking': 6810, 'Linguistic Probing': 47482, 'Answer Verification': 15195, 'Gender Classification': 19119, 'Punctuation Error Detection': 100, 'Sentence Compression': 4934, 'Text Completion': 46997, 'Entity Relation Classification': 5903, 'Style Transfer': 985, 'Entity Generation': 3095, 'Spam Classification': 1065, 'Information Extraction': 33053, 'Discourse Connective Identification': 1000, 'Stance Detection': 1693, 'Toxic Language Detection': 99702, 'Text to Code': 49441, 'Misc.': 65866, 'Sentence Expansion': 1761, 'Speaker Relation Classification': 153, 'Stereotype Detection': 17351, 'Program Execution': 433157}
    """
    category_selection = {
        "Question Understanding": 59954,
        "Word Relation Classification": 8872,
        "Commonsense Classification": 130257,
        "Text Categorization": 0.2,
        "Text Completion": 0.3,
        "Coherence Classification": 0.1,
    }
    final_datasets = []
    # prepare integer counts first
    int_categories = [x for x in category_selection if isinstance(category_selection[x], int)]
    float_categories = [x for x in category_selection if isinstance(category_selection[x], float)]
    total_ratio_floats = sum([category_selection[x] for x in float_categories])
    for category in int_categories:
        subset = [x for x in data if x['meta']['categories'][0] == category]
        subset = np.random.choice(subset, category_selection[category], replace=False).tolist()
        final_datasets.extend(subset)
    # remaining samples:
    remaining_sample_size = desired_size - len(final_datasets)
    print(f"remaining_sample_size: {remaining_sample_size}")
    for category in float_categories:
        subset = [x for x in data if x['meta']['categories'][0] == category]
        subset_size = int(category_selection[category] / total_ratio_floats * remaining_sample_size)
        try:
            subset = np.random.choice(subset, subset_size, replace=False).tolist()
            final_datasets.extend(subset)
        except ValueError:
            print(f"category: {category}; desired subset_size: {subset_size}; len(subset): {len(subset)}")
            
    print(f"final_datasets: {len(final_datasets)}")
    # shuffle final_datasets
    # set np seed
    np.random.seed(42)
    np.random.shuffle(final_datasets)
    with open(args.output_jsonl, "w") as f:
        for conv in final_datasets:
            f.write(json.dumps(conv) + "\n")
    
if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-jsonl", type=str, default="tmp/datasets/prepared/ni_train.jsonl")
    parser.add_argument("--portion", type=float, default=0.1)
    parser.add_argument("--output-jsonl", type=str, default="tmp/datasets/prepared/ni_train_selected.jsonl")
    select(parser.parse_args())