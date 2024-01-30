import os
import json
from src.engine.datasets import MixteraDataset

# function returns: line_id -> property
def process_rpj_meta(dataset_path):
    def extract(example):
        if "language" in example and len(example["language"])>0:
            return f"language={example['language'][0]['name']}"
        elif "language" in example and len(example["language"])==0:
            return "language=unknown"
        elif "publication_date" in example:
            return f"publication_date={example['publication_date']}"
        else:
            return "unknown"
    files = [x for x in os.listdir(dataset_path) if x.endswith(".jsonl")]
    for file in files:
        with open(os.path.join(dataset_path, file)) as f:
            for line_id, line in enumerate(f):
                yield file, line_id, extract(json.loads(line)['meta'])

mds = MixteraDataset.from_folder(".cache/datasets/rpj_sample")
mds.prepare(process_fn=process_rpj_meta)