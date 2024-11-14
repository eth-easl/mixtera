#pragma once

#include <pybind11/pybind11.h>
#include <arrow/api.h>
#include <arrow/python/pyarrow.h>
#include <arrow/type_traits.h>
#include <arrow/array.h>

#include <unordered_map>
#include <vector>
#include <string>

namespace py = pybind11;

// Define MixtureKey equivalent in C++
struct MixtureKeyCpp {
    std::unordered_map<std::string, std::vector<std::string>> properties;

    bool operator==(const MixtureKeyCpp& other) const {
        return properties == other.properties;
    }
};

// Hash function for MixtureKeyCpp
namespace std {
    template <>
    struct hash<MixtureKeyCpp> {
        std::size_t operator()(const MixtureKeyCpp& key) const {
            std::size_t seed = 0;
            for (const auto& [prop_name, prop_values] : key.properties) {
                seed ^= std::hash<std::string>{}(prop_name);
                for (const auto& val : prop_values) {
                    seed ^= std::hash<std::string>{}(val);
                }
            }
            return seed;
        }
    };
}

// Define the types for the chunker index data
using Interval = std::pair<int64_t, int64_t>;
using FileIntervals = std::unordered_map<int64_t, std::vector<Interval>>;
using DatasetFiles = std::unordered_map<int64_t, FileIntervals>;
using ChunkerIndexCpp = std::unordered_map<MixtureKeyCpp, DatasetFiles>;

// Function declarations
py::object create_chunker_index(py::object py_table, int num_threads);

// Function to merge two sorted vectors of intervals
void merge_sorted_intervals_inplace(std::vector<Interval>& target_intervals,
                                      std::vector<Interval>& source_intervals);
                                      
// Function to merge per-thread chunker indices
void merge_chunker_indices(std::vector<ChunkerIndexCpp>& thread_indices,
                           ChunkerIndexCpp& merged_index);

void process_batch(const std::shared_ptr<arrow::RecordBatch>& batch,
                   const std::vector<std::string>& property_columns,
                   ChunkerIndexCpp& local_chunker_index);

bool GetIndexValue(const std::shared_ptr<arrow::Array>& indices, int64_t position, int64_t& out_index);