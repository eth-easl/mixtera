#pragma once

#include <arrow/api.h>
#include <arrow/array.h>
#include <arrow/python/pyarrow.h>
#include <arrow/type_traits.h>
#include <pybind11/pybind11.h>
#include <spdlog/spdlog.h>

#include <string>
#include <unordered_map>
#include <vector>

#include "absl/container/flat_hash_map.h"

#define FAIL(msg)                                                                                            \
  throw mixtera::utils::MixteraException("ERROR at " __FILE__ ":" + std::to_string(__LINE__) + " " + (msg) + \
                                         "\nExecution failed.")

#define ASSERT(expr, msg)         \
  if (!static_cast<bool>(expr)) { \
    FAIL((msg));                  \
  }                               \
  static_assert(true, "End call of macro with a semicolon")

#ifdef NDEBUG
#define DEBUG_ASSERT(expr, msg) \
  do {                          \
  } while (0)
#else
#define DEBUG_ASSERT(expr, msg) ASSERT((expr), (msg))
#endif

namespace mixtera::utils {

class MixteraException : public std::exception {
 public:
  explicit MixteraException(std::string msg) : msg_{std::move(msg)} {}
  const char* what() const noexcept override { return msg_.c_str(); }

 private:
  const std::string msg_;
};

}  // namespace mixtera::utils

namespace py = pybind11;

template <typename T1, typename T2>
using MapT = std::map<T1, T2>;

using MixtureKeyCpp = std::string;
using Interval = std::pair<int64_t, int64_t>;
using FileIntervals = MapT<int64_t, std::vector<Interval>>;
using DatasetFiles = MapT<int64_t, FileIntervals>;
using ChunkerIndexCpp = MapT<MixtureKeyCpp, DatasetFiles>;

// Main function
py::object create_chunker_index(py::object py_table, int num_threads);

// Helper functions
std::vector<std::string> fetch_property_columns(const arrow::Table& table);
std::vector<ChunkerIndexCpp> calc_thread_chunker_indices(py::object& py_table, uint32_t num_threads);
ChunkerIndexCpp merge_chunker_indices(std::vector<ChunkerIndexCpp>* thread_chunker_indices);
ChunkerIndexCpp merge_chunker_indices_impl(std::vector<ChunkerIndexCpp>* thread_chunker_indices);
py::dict string_key_to_property_dict(const MixtureKeyCpp& key);
py::object build_py_chunker_index(ChunkerIndexCpp* merged_chunker_index);

void merge_sorted_intervals_inplace(std::vector<Interval>& target_intervals, std::vector<Interval>& source_intervals);

void process_batch(const std::shared_ptr<arrow::RecordBatch>& batch, const std::vector<std::string>& property_columns,
                   ChunkerIndexCpp& local_chunker_index);

bool GetIndexValue(const std::shared_ptr<arrow::Array>& indices, int64_t position, int64_t& out_index);