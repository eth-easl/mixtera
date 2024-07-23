import json

with open(".cache/meta/natural_instructions/train_tasks.txt", "r") as f:
    train_tasks = [line.strip().split("_")[0] for line in f]
with open(".cache/meta/natural_instructions/test_tasks.txt", "r") as f:
    test_tasks = [line.strip().split("_")[0] for line in f]
with open(".cache/meta/natural_instructions/excluded_tasks.txt", "r") as f:
    excluded_tasks = [line.strip().split("_")[0] for line in f]

with open(".cache/prepared/natural_instructions.jsonl", "r") as f:
    data = [json.loads(line) for line in f]
# remove excluded tasks
print(f"Before: {len(data)}")
data = [x for x in data if x['meta']['task_id'] not in excluded_tasks]
print(f"After removing excluded tasks: {len(data)}")
# split into train and test
train_data = [x for x in data if x['meta']['task_id'] in train_tasks]
test_data = [x for x in data if x['meta']['task_id'] in test_tasks]
print(f"Train: {len(train_data)}")
print(f"Test: {len(test_data)}")

with open(".cache/prepared/ni_train.jsonl", "w") as f:
    for conv in train_data:
        f.write(json.dumps(conv) + "\n")
        
with open(".cache/prepared/ni_test.jsonl", "w") as f:
    for conv in test_data:
        f.write(json.dumps(conv) + "\n")