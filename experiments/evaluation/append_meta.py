import json
output = "tmp/experiments/gemma2b-it.random.step2000.jsonl"
with open("tmp/datasets/prepared/ni_test.jsonl", "r") as f:
    test_data = [json.loads(line) for line in f]

with open(output, "r") as f:
    results = [json.loads(line) for line in f]

for i in range(len(results)):
    results[i]['meta'] = test_data[i]['meta']

with open(output, 'w') as f:
    for result in results:
        f.write(json.dumps(result) + "\n")