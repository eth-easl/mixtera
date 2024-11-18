import json

def evaluate(args):
    with open(args.eval_data, "r") as f:
        eval_data = [json.loads(line) for line in f]
    total_size = len(eval_data)
    is_correct = [1 if d['output']==d['ground_truth'] else 0 for d in eval_data]
    correct = sum(is_correct)
    print(f"Overall Accuracy: {correct/total_size}, correct: {correct}, total: {total_size}")
    result = {
        'meta': {
            'filename': args.eval_data,
        },
        'overall': {
            'correct': correct,
            'total': total_size,
            'accuracy': correct/total_size
        },
        'per_task': {},
        'per_domain': {}
    }
    # per task accuracy
    tasks = set([d['meta']['task_id'] for d in eval_data])
    for task in tasks:
        task_data = [d for d in eval_data if d['meta']['task_id']==task]
        task_size = len(task_data)
        task_correct = sum([1 if d['output']==d['ground_truth'] else 0 for d in task_data])
        result['per_task'][task] = {
            'correct': task_correct,
            'total': task_size,
            'accuracy': task_correct/task_size
        }
    # per category accuracy
    categories = []
    domains = []
    for d in eval_data:
        categories.extend(d['meta']['categories'])
        domains.extend(d['meta']['domain'])
    categories = set(categories)
    domains = set(domains)
    for category in categories:
        category_data = [d for d in eval_data if category in d['meta']['categories']]
        category_size = len(category_data)
        category_correct = sum([1 if d['output']==d['ground_truth'] else 0 for d in category_data])
        result['per_task'][category] = {
            'correct': category_correct,
            'total': category_size,
            'accuracy': category_correct/category_size
        }
    for domain in domains:
        domain_data = [d for d in eval_data if domain in d['meta']['domain']]
        domain_size = len(domain_data)
        domain_correct = sum([1 if d['output']==d['ground_truth'] else 0 for d in domain_data])
        result['per_domain'][domain] = {
            'correct': domain_correct,
            'total': domain_size,
            'accuracy': domain_correct/domain_size
        }
    
    with open(args.output, 'w') as f:
        f.write(json.dumps(result))

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Score the NI model')
    parser.add_argument('--eval-data', type=str, help='Path to the output file after running eval_ni.py')
    parser.add_argument('--output', type=str, help='Path to the output')
    evaluate(parser.parse_args())