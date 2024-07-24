import os
import json
from tqdm import tqdm
from protocol import Conversation

data_path = ".cache/datasets/natural-instructions/tasks"
tasks = [x for x in os.listdir(data_path) if x.endswith(".json")]
conversations = []

for task in tqdm(tasks):
    filepath = os.path.join(data_path, task)
    with open(filepath, "r") as f:
        data = json.load(f)
    task_id = task.split("_")[0]
    instances = data['Instances']
    meta = {
        "task_id": task_id,
        "categories": data['Categories'],
        "reasoning": data['Reasoning'],
        "language": {
            "input": data['Input_language'],
            "output": data['Output_language']
        },
        "domain": data['Domains']
    }
    user_instruction = data['Definition'][0]
    for inst in instances:
        conv = Conversation()
        user_prompt = f"{user_instruction}\n{inst['input']}"
        conv.append("user", user_prompt)
        conv.append("assistant", inst['output'][0])
        conv.add_meta(meta)
        conversations.append(conv)

os.makedirs(".cache/datasets/prepared", exist_ok=True)
with open(".cache/datasets/prepared/natural_instructions.jsonl", "w") as f:
    for conv in conversations:
        conv.sanity_check() # this ensures no repeated user prompts
        f.write(conv.to_json() + "\n")