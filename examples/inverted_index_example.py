import portion as P

from mixtera.core.datacollection.index.index_collection import create_inverted_index_interval_dict
from mixtera.utils.utils import merge_property_dicts

# This example code is based on the QueryResult::_invert_result method. It essentially unrolls the for loops to
# detail what operations take place within it.

# Mixtera conventionally stores an index that points from dataset to file to property to value to ranges of rows. This
# layout is useful for running queries, but becomes cumbersome when trying to create chunks from a query result for two
# reasons: (1) we would ideally index row ranges by property combinations and (2) the mixtera index has duplicates
# across properties and values - e.g. a row can contain both English and French data - hence appearing in several of the
# index's entries. As a consequence, we want to eventually transform the result of a query - which is a conventional
# index - into another index - called a chunker index - that has two important qualities: (1) it allows us to index
# row ranges by concrete property-value combinations which are required for a specific mixture and (2) it deduplicates
# the data, which ensures at-most-once visitation guarantees are maintained.

# As part of the chunker index construction, we generate an inverted index with the following structure:
# {
#     "dataset_id": {
#         "file_id": {
#             portion.Interval: {
#                 feature_1_name: [feature_1_value_0, ...],
#                 ...
#                 feature_n_name: [feature_n_value_0, ...],
#             },
#             ...
#         },
#         ...
#     },
#     ...
# }

# The inverted index turns the conventional index on its head, by moving properties and their values to the last level,
# and moving dataset and file ids at the first level. Importantly, the inverted index, takes care of the set algebra
# required in order to split the conventional index's cross property row ranges into disjoint intervals to which common
# properties and property values are merged and assigned. From this we can eventually build a chunker index.

# To construct the inverted index, we use Portion which is a library that facilitates set theory algebra.

# Besides defining different types of intervals (e.g. '[x, y)' would correspond to portion.closedopen(x, y)), Portion
# defines the IntervalDict data structure that has been designed to point from intervals to arbitrary values. This is
# the foundation of our inverted index.

# Let's define the basic data structures
inverted_index = create_inverted_index_interval_dict()

raw_index = {
    "language": {
        "english": {
            0: {0: [(50, 150)]},
        },
        "french": {
            0: {
                0: [(0, 100), (150, 200)]
            }
        },
        "german": {
            0: {
                0: [(100, 250)]
            }
        }
    },
}


# Let's define a shorthand helper method that creates [x, y) intervals in portion for indexing using our raw_index
def make_key(property_name, property_value, file_id, range_idx, dataset_id=0):
    raw_interval = raw_index[property_name][property_value][dataset_id][file_id][range_idx]
    return P.closedopen(*raw_interval)


# Let's start building the inverted index

## Let's create our first interval, which will implicitly serve as a key
current_key = make_key("language", "english", 0, 0)
print(" 1>", current_key)  # This should print '[50,150)'

# To this index, we assign the property and property value we are currently at (language:english)
inverted_index[0][0][current_key] = {"language": ["english"]}
print(" 2>", inverted_index)


## Now that we added that property-value pair, let's move on to the next, which is 'language:french'

# We start with file 0
current_key = make_key("language", "french", 0, 0)
print(" 3>", current_key)  # This should print '[0,100)'

# Here is where the magic of portion and the interval dict comes into play. We can use this interval to index into the
# IntervalDict. The IntervalDict will automatically break the interval up into the intersecting intervals.
intersections = inverted_index[0][0][current_key]
# Because [50, 150) ∩ [0, 100) = [50, 100)  --> this should print '{[50,100): {'language': ['english']}}'
print(" 4>", intersections)

# So what happens when, like the previous time, we try to add a new property-value to the inverted index?
inverted_index[0][0][current_key] = {"language": ["french"]}

# You'll notice that the behavior changes this time: the intersected part ([50, 100)) disappears and gets overwritten by
# the current_key interval ([0, 100)) with "language:french". The non-intersected part remains ([100, 150)) and retains
# its "language:english" property-value pair.
print(" 5>", inverted_index)

# However we still have the intersected interval ([50, 100)) with its old "language:english" property-pair to consider

# What we will do here is the following. We leverage the intersections we recorded a while back. This time round,
# only [50, 100) was an intersection. Each iteration in the loop will produce an (intersection_interval, old_properties)
# pair. Recall that indexing an IntervalDict with an interval will produce a dict of intersecting intervals and their
# associated values. What we do here is fairly straight forward. We overwrite intersection intervals with the merged
# property-value dictionaries.
for interval, intersection_properties in intersections.items():
    print(" 6.1>", interval)
    print(" 6.2>", inverted_index[0][0][interval].values()[0])
    print(" 6.3>", intersection_properties)
    inverted_index[0][0][interval] = merge_property_dicts(
        inverted_index[0][0][interval].values()[0], intersection_properties
    )
    print(" 6.4>", inverted_index[0][0][interval])
print(" 6.5>", inverted_index)

# Next we have the case when there is not intersection. This is pretty straight forward and essentially the intersection
# loop never needs to be executed

current_key = make_key("language", "french", 0, 1)
print(" 7>", current_key)  # This should print '[150,200)'

intersections = inverted_index[0][0][current_key]
# Because ([0, 50) ∪ [50, 100) ∪ [100, 150)) ∩ [150, 200) = ∅
print(" 8>", intersections)

# Note that for (did, fid) = (0, 0) you get '{[0,50) | [150,200): {'language': ['french']}'. This is a Portion interval
# shorthand that basically denotes [0,50) ∪ [150,200), and is an iterable structure (i.e. you iterate over
# [[0,50), [150,200)]). This is important for our next example where in the intersection loop we'll have to update
# several intersection intervals.
inverted_index[0][0][current_key] = {"language": ["french"]}
print(" 9>", inverted_index)

# We move to the final case, where we intersect (and thus need to update) several intervals within a file

current_key = make_key("language", "german", 0, 0)
print("10>", current_key)  # This should print '[100,250)'

intersections = inverted_index[0][0][current_key]
# Because ([0, 50) ∪ [50, 100) ∪ [100, 150) ∪ [150, 200)) ∩ [100, 250) = [100, 150) ∪ [150, 200)
print("11>", intersections)

# We overwrite the [100, 250) interval with german
inverted_index[0][0][current_key] = {"language": ["german"]}
print("12>", inverted_index)

# Then we update all intersections such that german is added to them as well
for idx, (interval, intersection_properties) in enumerate(intersections.items()):
    print(f"{13 + idx}.1>", interval)
    print(f"{13 + idx}.2>", inverted_index[0][0][interval].values()[0])
    print(f"{13 + idx}.3>", intersection_properties)
    inverted_index[0][0][interval] = merge_property_dicts(
        inverted_index[0][0][interval].values()[0], intersection_properties
    )
    print(f"{13 + idx}.4>", inverted_index[0][0][interval])
print("15>", inverted_index)

# The important things to note is how all intervals are disjoint within a file, and that each interval has an associated
# dictionary that identifies what properties it owns and the values of that property that apply to the interval.
