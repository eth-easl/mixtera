#pragma once

#include <arrow/api.h>
#include <arrow/array.h>
#include <arrow/python/pyarrow.h>
#include <arrow/type_traits.h>
#include <pybind11/pybind11.h>

#include <string>
#include <unordered_map>
#include <vector>

#include "absl/container/flat_hash_map.h"

namespace py = pybind11;

using MixtureKeyCpp = std::string;
using Interval = std::pair<int64_t, int64_t>;
using FileIntervals = absl::flat_hash_map<int64_t, std::vector<Interval>>;
using DatasetFiles = absl::flat_hash_map<int64_t, FileIntervals>;
using ChunkerIndexCpp = absl::flat_hash_map<MixtureKeyCpp, DatasetFiles>;

// Main function
py::object create_chunker_index(py::object py_table, int num_threads);

// Helper functions
std::vector<std::string> fetch_property_columns(const arrow::Table& table);

// Function to merge two sorted vectors of intervals
void merge_sorted_intervals_inplace(std::vector<Interval>& target_intervals, std::vector<Interval>& source_intervals);

// Function to merge per-thread chunker indices
void merge_chunker_indices(std::vector<ChunkerIndexCpp>& thread_indices, ChunkerIndexCpp& merged_index);

void process_batch(const std::shared_ptr<arrow::RecordBatch>& batch, const std::vector<std::string>& property_columns,
                   ChunkerIndexCpp& local_chunker_index);

bool GetIndexValue(const std::shared_ptr<arrow::Array>& indices, int64_t position, int64_t& out_index);