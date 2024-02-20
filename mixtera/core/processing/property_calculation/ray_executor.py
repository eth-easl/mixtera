# just notes, to be implemented


# We have an actor class
# setup func called in init, can init model etc - gets reference to the class itself
# load data creates a ray dataset from all the files
# run uses map_batches with the according batch size and parallelism to get one prediction per datapoint per batch
# for categorical values, we can then use the groupby/aggregate logic as below (with bucket: (file_id, line_id))
# for numerical values, we need another map operation (not supplied by user) to generate buckets
# we can then rangeify per file id (maybe in ray or just in python, not sure yet)
# then we return.

# from ray.data.aggregate import AggregateFn

# def aggregate_names(names: List[str], batch: Dict[str, np.ndarray]) -> List[str]:
#    return names + batch["name"].tolist()

# ds = (ray.data.from_items([
#        {"name": "Luna", "age": 4},
#        {"name": "Rory", "age": 14},
#        {"name": "Scout", "age": 4},
#        {"name": "Maxi", "age": 14},
#        {"name": "Maxi", "age": 6}
#    ])
#    .map_batches(add_dog_years, batch_size=2)
#    .groupby("age")
#    .aggregate(AggregateFn(init=lambda: [], accumulate=aggregate_names, merge=lambda x, y: x + y))
# )

# for age, names in ds.iter_rows():
#    print(f"Age: {age}, Names: {names}")
