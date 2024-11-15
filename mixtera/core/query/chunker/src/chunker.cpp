// chunker.cpp
#include "chunker.hpp"

#include <arrow/api.h>
#include <arrow/python/pyarrow.h>
#include <arrow/util/checked_cast.h>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <spdlog/spdlog.h>

#include <indicators/block_progress_bar.hpp>
#include <indicators/cursor_control.hpp>
#include <indicators/progress_bar.hpp>
#include <indicators/termcolor.hpp>
#include <iostream>
#include <thread>
#include <typeinfo>

#include "absl/container/flat_hash_set.h"

namespace py = pybind11;

bool GetIndexValue(const std::shared_ptr<arrow::Array>& indices, int64_t position, int64_t& out_index) {
  if (position < 0 || position >= indices->length()) {
    FAIL(fmt::format("Index position out of bounds: {}", position));
  }

  switch (indices->type_id()) {
    case arrow::Type::INT8: {
      auto index_array = arrow::internal::checked_pointer_cast<arrow::Int8Array>(indices);
      out_index = static_cast<int64_t>(index_array->Value(position));
      return true;
    }
    case arrow::Type::UINT8: {
      auto index_array = arrow::internal::checked_pointer_cast<arrow::UInt8Array>(indices);
      out_index = static_cast<int64_t>(index_array->Value(position));
      return true;
    }
    case arrow::Type::INT16: {
      auto index_array = arrow::internal::checked_pointer_cast<arrow::Int16Array>(indices);
      out_index = static_cast<int64_t>(index_array->Value(position));
      return true;
    }
    case arrow::Type::UINT16: {
      auto index_array = arrow::internal::checked_pointer_cast<arrow::UInt16Array>(indices);
      out_index = static_cast<int64_t>(index_array->Value(position));
      return true;
    }
    case arrow::Type::INT32: {
      auto index_array = arrow::internal::checked_pointer_cast<arrow::Int32Array>(indices);
      out_index = static_cast<int64_t>(index_array->Value(position));
      return true;
    }
    case arrow::Type::UINT32: {
      auto index_array = arrow::internal::checked_pointer_cast<arrow::UInt32Array>(indices);
      out_index = static_cast<int64_t>(index_array->Value(position));
      return true;
    }
    case arrow::Type::INT64: {
      auto index_array = arrow::internal::checked_pointer_cast<arrow::Int64Array>(indices);
      out_index = index_array->Value(position);
      return true;
    }
    case arrow::Type::UINT64: {
      auto index_array = arrow::internal::checked_pointer_cast<arrow::UInt64Array>(indices);
      out_index = static_cast<int64_t>(index_array->Value(position));
      return true;
    }
    default:
      spdlog::error(fmt::format("Unsupported index type: {}", indices->type()->ToString()));
      return false;
  }
}

void merge_sorted_intervals_inplace(std::vector<Interval>& target_intervals, std::vector<Interval>& source_intervals) {
  const uint64_t m = target_intervals.size();
  const uint64_t n = source_intervals.size();

  target_intervals.reserve(m + n);

  // Copy source_intervals to target_intervals
  target_intervals.insert(target_intervals.end(), std::make_move_iterator(source_intervals.begin()),
                          std::make_move_iterator(source_intervals.end()));

  source_intervals.clear();
  source_intervals.shrink_to_fit();  // Free memory

  std::inplace_merge(target_intervals.begin(), target_intervals.begin() + m, target_intervals.end());
}

// Helper function to process a range of rows
void process_batch(const std::shared_ptr<arrow::RecordBatch>& batch, const std::vector<std::string>& property_columns,
                   ChunkerIndexCpp& local_chunker_index) {
  try {
    const uint64_t num_rows = batch->num_rows();

    // Get required columns
    auto dataset_id_array = batch->GetColumnByName("dataset_id");
    auto file_id_array = batch->GetColumnByName("file_id");
    auto interval_start_array = batch->GetColumnByName("interval_start");
    auto interval_end_array = batch->GetColumnByName("interval_end");

    if (!dataset_id_array || !file_id_array || !interval_start_array || !interval_end_array) {
      FAIL("One or more required columns are missing in batch.");
    }

    // Prepare property arrays
    std::vector<std::shared_ptr<arrow::Array>> property_arrays;
    for (const std::string& col_name : property_columns) {
      auto array = batch->GetColumnByName(col_name);
      if (!array) {
        FAIL(fmt::format("Property column {} not found in batch.", col_name));
      }
      property_arrays.push_back(array);
    }

    for (uint64_t i = 0; i < num_rows; ++i) {
      std::string key;

      bool first_prop = true;
      for (size_t j = 0; j < property_columns.size(); ++j) {
        auto array = property_arrays[j];
        const std::string& col_name = property_columns[j];

        if (!array || array->IsNull(i)) {
          continue;
        }

        if (!first_prop) {
          key += ";";
        }
        first_prop = false;

        key += col_name + ":";

        auto type_id = array->type_id();

        try {
          // Handle STRING and LARGE_STRING types
          if (type_id == arrow::Type::STRING || type_id == arrow::Type::LARGE_STRING) {
            std::string value;
            if (type_id == arrow::Type::STRING) {
              auto str_array = arrow::internal::checked_pointer_cast<arrow::StringArray>(array);
              value = str_array->GetString(i);
            } else {
              auto str_array = arrow::internal::checked_pointer_cast<arrow::LargeStringArray>(array);
              value = str_array->GetString(i);
            }
            key += value;

          // Handle LIST and LARGE_LIST types
          } else if (type_id == arrow::Type::LIST || type_id == arrow::Type::LARGE_LIST) {
            std::shared_ptr<arrow::Array> value_array;
            int64_t offset = 0;
            int64_t length = 0;

            if (type_id == arrow::Type::LIST) {
              auto list_array = arrow::internal::checked_pointer_cast<arrow::ListArray>(array);
              if (list_array->IsNull(i)) {
                continue;
              }
              offset = list_array->value_offset(i);
              length = list_array->value_length(i);
              value_array = list_array->values();
            } else {
              auto list_array = arrow::internal::checked_pointer_cast<arrow::LargeListArray>(array);
              if (list_array->IsNull(i)) {
                continue;
              }
              offset = list_array->value_offset(i);
              length = list_array->value_length(i);
              value_array = list_array->values();
            }

            if (!value_array) {
              FAIL(fmt::format("Value array is null for LIST column '{}'", col_name));
            }

            if (offset < 0 || length < 0 || (offset + length) > value_array->length()) {
              FAIL(fmt::format("Invalid offset/length for LIST column '{}' at row {}", col_name, i));
            }

            // Build values string
            std::string values_str;
            bool first_value = true;

            const auto value_type_id = value_array->type_id();

            // Handle STRING and LARGE_STRING elements within the list
            if (value_type_id == arrow::Type::STRING || value_type_id == arrow::Type::LARGE_STRING) {
              for (int64_t k = offset; k < offset + length; ++k) {
                if (value_array->IsNull(k)) {
                  continue;
                }
                std::string val;
                if (value_type_id == arrow::Type::STRING) {
                  auto str_values = arrow::internal::checked_pointer_cast<arrow::StringArray>(value_array);
                  val = str_values->GetString(k);
                } else {
                  auto str_values = arrow::internal::checked_pointer_cast<arrow::LargeStringArray>(value_array);
                  val = str_values->GetString(k);
                }
                if (!first_value) {
                  values_str += ",";
                }
                first_value = false;
                values_str += val;
              }

            // Handle DICTIONARY elements within the list
            } else if (value_type_id == arrow::Type::DICTIONARY) {
              auto dict_array = arrow::internal::checked_pointer_cast<arrow::DictionaryArray>(value_array);
              const auto& dict = dict_array->dictionary();
              const auto& indices = dict_array->indices();

              if (!dict || !indices) {
                FAIL(fmt::format("Dictionary values or indices are null in LIST of DICTIONARY for column '{}'",
                                 col_name));
              }

              for (int64_t k = offset; k < offset + length; ++k) {
                if (dict_array->IsNull(k)) {
                  continue;
                }

                int64_t index = 0;
                if (!GetIndexValue(indices, k, index)) {
                  FAIL(fmt::format("Failed to get index value at position {} in LIST of DICTIONARY for column {}", k,
                                   col_name));
                }

                if (index < 0 || index >= dict->length()) {
                  FAIL(fmt::format("Index out of bounds in dictionary values for LIST of DICTIONARY at position {} ",
                                   k));
                }

                std::string val;
                if (dict->type_id() == arrow::Type::STRING) {
                  auto dict_values = arrow::internal::checked_pointer_cast<arrow::StringArray>(dict);
                  val = dict_values->GetString(index);
                } else if (dict->type_id() == arrow::Type::LARGE_STRING) {
                  auto dict_values = arrow::internal::checked_pointer_cast<arrow::LargeStringArray>(dict);
                  val = dict_values->GetString(index);
                } else {
                  FAIL(fmt::format("Unsupported dictionary value type in LIST of DICTIONARY for column '{}'. Type: {}", col_name,
                                   dict->type()->ToString()));
                }

                if (!first_value) {
                  values_str += ",";
                }
                first_value = false;
                values_str += val;
              }

            } else {
              FAIL(fmt::format("Unsupported list element type in column '{}'. Type: {}", col_name,
                               value_array->type()->ToString()));
            }

            key += values_str;

          // Handle DICTIONARY type
          } else if (type_id == arrow::Type::DICTIONARY) {
            auto dict_array = arrow::internal::checked_pointer_cast<arrow::DictionaryArray>(array);

            if (dict_array->IsNull(i)) {
              continue;
            }

            const auto& dict = dict_array->dictionary();
            const auto& indices = dict_array->indices();

            if (!dict || !indices) {
              FAIL(fmt::format("Dictionary values or indices are null in DICTIONARY column '{}'", col_name));
            }

            int64_t index = 0;
            if (!GetIndexValue(indices, i, index)) {
              FAIL(fmt::format("Failed to get index value at position {}  in DICTIONARY column '{}'", i, col_name));
            }

            if (index < 0 || index >= dict->length()) {
              FAIL(fmt::format("Index out of bounds in dictionary values for DICTIONARY at position {}", i));
            }

            std::string value;
            if (dict->type_id() == arrow::Type::STRING) {
              auto dict_values = arrow::internal::checked_pointer_cast<arrow::StringArray>(dict);
              value = dict_values->GetString(index);
            } else if (dict->type_id() == arrow::Type::LARGE_STRING) {
              auto dict_values = arrow::internal::checked_pointer_cast<arrow::LargeStringArray>(dict);
              value = dict_values->GetString(index);
            } else {
              FAIL(fmt::format("Unsupported dictionary value type in DICTIONARY column '{}'. Type: {}", col_name,
                               dict->type()->ToString()));
            }

            key += value;

          } else {
            FAIL(fmt::format("Unsupported array type in column '{}' at row {}. Type: {}", col_name, i,
                             array->type()->ToString()));
          }
        } catch (const std::exception& e) {
          spdlog::error(fmt::format("Exception at row {}, column {}: {}", i, col_name, e.what()));
          throw;
        }
      }

      // Retrieve interval data and store it
      int64_t dataset_id = 0;
      if (!GetIndexValue(dataset_id_array, i, dataset_id)) {
        FAIL(fmt::format("Failed to get dataset_id at row {}", i));
      }

      int64_t file_id = 0;
      if (!GetIndexValue(file_id_array, i, file_id)) {
        FAIL(fmt::format("Failed to get file_id at row {}", i));
      }

      int64_t interval_start = 0;
      if (!GetIndexValue(interval_start_array, i, interval_start)) {
        FAIL(fmt::format("Failed to get interval_start at row {}", i));
      }

      int64_t interval_end = 0;
      if (!GetIndexValue(interval_end_array, i, interval_end)) {
        FAIL(fmt::format("Failed to get interval_end at row {}", i));
      }

      if (interval_end < interval_start) {
        FAIL(fmt::format("interval_end = {} < interval_start = {} at row {} (file {}, dataset {})", interval_end,
                         interval_start, i, file_id, dataset_id));
      }

      spdlog::info(fmt::format("key {} dataset {} file {} interval start {} interval end {}", key, dataset_id, file_id, interval_start, interval_end));

      Interval interval = {interval_start, interval_end};

      // Store intervals in the local chunker index
      local_chunker_index[key][dataset_id][file_id].push_back(interval);
    }

  } catch (const std::exception& e) {
    spdlog::error(fmt::format("Exception in process_rows: {}", e.what()));
    throw;
  }
}

std::vector<std::string> fetch_property_columns(const arrow::Table& table) {
  // Pre-defined keys that are not property columns
  absl::flat_hash_set<std::string> exclude_keys = {"dataset_id", "file_id", "group_id", "interval_start",
                                                   "interval_end"};
  std::vector<std::string> result;

  for (const auto& field : table.schema()->fields()) {
    const std::string& col_name = field->name();
    if (!exclude_keys.contains(col_name)) {
      result.push_back(col_name);
    }
  }

  return result;
}

std::vector<ChunkerIndexCpp> calc_thread_chunker_indices(py::object& py_table, uint32_t num_threads) {
  arrow::py::import_pyarrow();  // Initialize Arrow C++ and Python bridges
  std::shared_ptr<arrow::Table> table = arrow::py::unwrap_table(py_table.ptr()).ValueOrDie();

  py::gil_scoped_release
      release;  // Release GIL for multithreading since now we do not operate on Python objects anymore

  // State setup
  const int64_t total_rows = table->num_rows();
  std::atomic<int64_t> total_rows_processed{0};
  const std::vector<std::string> property_columns = fetch_property_columns(*table);
  arrow::TableBatchReader batch_reader(*table);
  std::vector<ChunkerIndexCpp> thread_chunker_indices(num_threads);
  std::atomic_bool exception_occurred{false};
  std::exception_ptr thread_exception = nullptr;
  std::mutex exception_mutex;

  // Read all record batches
  std::vector<std::shared_ptr<arrow::RecordBatch>> batches;
  std::shared_ptr<arrow::RecordBatch> batch;
  while (batch_reader.ReadNext(&batch).ok() && batch) {
    batches.push_back(batch);
  }

  // Initialize the progress bar
  indicators::BlockProgressBar progress_bar{
      indicators::option::BarWidth{50},
      indicators::option::Start{"["},
      indicators::option::End{"]"},
      indicators::option::ForegroundColor{indicators::Color::green},
      indicators::option::PrefixText{"Processing rows: "},
      indicators::option::ShowElapsedTime{true},
      indicators::option::ShowRemainingTime{true},
      indicators::option::MaxProgress{static_cast<size_t>(total_rows)},
      indicators::option::Stream{std::cout},  // Specify the output stream
      indicators::option::FontStyles{std::vector<indicators::FontStyle>{indicators::FontStyle::bold}}};

  // Determine batch ranges for each thread
  const uint64_t num_batches = batches.size();
  const uint64_t batches_per_thread = (num_batches + num_threads - 1) / num_threads;

  // Launch threads
  spdlog::info("Spawning {} threads to processes {} batches.", num_threads, num_batches);
  std::vector<std::thread> threads;
  for (uint32_t t = 0; t < num_threads; ++t) {
    int64_t start_batch = t * batches_per_thread;
    int64_t end_batch = std::min(start_batch + batches_per_thread, num_batches);
    if (start_batch < end_batch) {
      threads.emplace_back([&, start_batch, end_batch, t]() {
        try {
          for (int64_t b = start_batch; b < end_batch; ++b) {
            process_batch(batches[b], property_columns, thread_chunker_indices[t]);
            const int64_t num_rows = batches[b]->num_rows();
            total_rows_processed += num_rows;
            progress_bar.set_progress(static_cast<size_t>(total_rows_processed.load()));
            // After processing, release the batch
            batches[b].reset();
          }
        } catch (const std::exception& e) {
          spdlog::error("Exception in thread {}: {}", t, e.what());
          {
            std::lock_guard<std::mutex> lock(exception_mutex);
            if (!exception_occurred.exchange(true)) {
              thread_exception = std::current_exception();
            }
          }
        }
      });
    }
  }

  // Wait for threads to finish
  for (auto& thread : threads) {
    thread.join();
  }

  if (thread_exception) {
    spdlog::error("A thread encountered an error, rethrowing it and propagating to Python.");
    std::rethrow_exception(thread_exception);
  }

  spdlog::info("All threads finished, clearing batches.");

  batches.clear();
  batches.shrink_to_fit();
  table.reset();

  py::gil_scoped_acquire acquire;  // Just for cleanliness, always acquire GIL at end of C++ scope.
  spdlog::info("calc_thread_chunker_indices done, GIL reacquired.");
  return thread_chunker_indices;
}

ChunkerIndexCpp merge_chunker_indices_impl(std::vector<ChunkerIndexCpp>* thread_chunker_indices) {
  ChunkerIndexCpp merged_chunker_index;

  const uint32_t total_indices = thread_chunker_indices->size();

  indicators::BlockProgressBar merge_bar{
      indicators::option::BarWidth{50},
      indicators::option::Start{"["},
      indicators::option::End{"]"},
      indicators::option::ForegroundColor{indicators::Color::yellow},
      indicators::option::PrefixText{"Merging indices: "},
      indicators::option::ShowElapsedTime{true},
      indicators::option::ShowRemainingTime{true},
      indicators::option::MaxProgress{total_indices},
      indicators::option::Stream{std::cout},
      indicators::option::FontStyles{std::vector<indicators::FontStyle>{indicators::FontStyle::bold}}};

  for (auto& local_index : *thread_chunker_indices) {
    for (auto& [key, datasets] : local_index) {
      auto& target_datasets = merged_chunker_index[key];

      for (auto& [dataset_id, files] : datasets) {
        auto& target_files = target_datasets[dataset_id];

        for (auto& [file_id, intervals] : files) {
          auto& target_intervals = target_files[file_id];

          if (target_intervals.empty()) {
            target_intervals = std::move(intervals);
          } else {
            merge_sorted_intervals_inplace(target_intervals, intervals);
          }
        }
        files.clear();            // Safe to clear after iteration
        files = FileIntervals();  // force deallocation by deleting the old object
      }
      datasets.clear();           // Safe to clear after iteration
      datasets = DatasetFiles();  // force deallocation
    }
    local_index.clear();              // Clear the local_index to free memory
    local_index = ChunkerIndexCpp();  // force deallocation

    merge_bar.tick();
  }

  // After processing all local indices, we can clear the thread_chunker_indices vector itself
  thread_chunker_indices->clear();
  thread_chunker_indices->shrink_to_fit();
  merge_bar.mark_as_completed();

  return merged_chunker_index;
}

ChunkerIndexCpp merge_chunker_indices(std::vector<ChunkerIndexCpp>* thread_chunker_indices) {
  py::gil_scoped_release release;
  spdlog::info("GIL released, merging chunker indices.");
  ChunkerIndexCpp merged_chunker_index;

  if (thread_chunker_indices->size() == 1) {
    merged_chunker_index = thread_chunker_indices->at(0);
  } else {
    merged_chunker_index = merge_chunker_indices_impl(thread_chunker_indices);
  }

  spdlog::info("Merging done, clearing memory.");
  thread_chunker_indices->clear();
  thread_chunker_indices->shrink_to_fit();

  py::gil_scoped_acquire acquire;  // Just for cleanliness, always acquire GIL at end of C++ scope.
  spdlog::info("merge_chunker_indices done, GIL reacquired.");
  return merged_chunker_index;
}

py::dict string_key_to_property_dict(const MixtureKeyCpp& key) {
  // Parse the C++ key string into a properties dictionary for Python
  // Expected format: "prop1:val1,val2;prop2:val3"
  py::dict py_properties;
  std::stringstream ss_props(key);
  std::string prop_pair;
  while (std::getline(ss_props, prop_pair, ';')) {
    size_t colon_pos = prop_pair.find(':');
    if (colon_pos == std::string::npos) {
      continue;
    }
    std::string prop_name = prop_pair.substr(0, colon_pos);
    std::string values_str = prop_pair.substr(colon_pos + 1);

    // Split values by ','
    std::vector<std::string> values;
    std::stringstream ss_values(values_str);
    std::string value;
    while (std::getline(ss_values, value, ',')) {
      values.push_back(value);
    }
    py_properties[py::cast(prop_name)] = py::cast(std::move(values));
  }
  return py_properties;
}

py::object build_py_chunker_index(ChunkerIndexCpp* merged_chunker_index) {
  const py::object mixture_module =
      py::module_::import("mixtera.core.query.mixture");  // Import the MixtureKey class from Python
  const py::object MixtureKey_class = mixture_module.attr("MixtureKey");
  py::dict py_chunker_index;

  const size_t total_keys = merged_chunker_index->size();
  const size_t update_interval =
      std::max<size_t>(std::ceil(static_cast<double>(total_keys) * 0.001), static_cast<size_t>(1));

  indicators::BlockProgressBar build_bar{
      indicators::option::BarWidth{50},
      indicators::option::Start{"["},
      indicators::option::End{"]"},
      indicators::option::ForegroundColor{indicators::Color::cyan},
      indicators::option::PrefixText{"Building Python object: "},
      indicators::option::ShowElapsedTime{true},
      indicators::option::ShowRemainingTime{true},
      indicators::option::MaxProgress{total_keys},
      indicators::option::Stream{std::cout},
      indicators::option::FontStyles{std::vector<indicators::FontStyle>{indicators::FontStyle::bold}}};

  uint64_t key_counter = 0;

  for (auto it = merged_chunker_index->begin(); it != merged_chunker_index->end();) {
    MixtureKeyCpp key = std::move(it->first);
    DatasetFiles datasets = std::move(it->second);

    auto current_it = it;
    ++it;
    merged_chunker_index->erase(current_it);  // We remove while iterating to reduce memory usage.

    py::object py_mixture_key = MixtureKey_class(string_key_to_property_dict(key));

    // The insertion order here is irrelevant. When we iterate over this in Python, we sort by keys.
    py::dict py_datasets;
    for (auto& [dataset_id, files] : datasets) {
      py::dict py_files;
      for (auto& [file_id, intervals] : files) {
        py_files[py::cast(file_id)] = py::cast(std::move(intervals));  // Convert vector of intervals to Python list
        intervals.clear();
        intervals.shrink_to_fit();
      }
      files.clear();
      files = FileIntervals();  // Force deallocation

      py_datasets[py::cast(dataset_id)] = py_files;
    }
    datasets.clear();
    datasets = DatasetFiles();  // Force deallocation

    py_chunker_index[py_mixture_key] = py_datasets;  // Add to chunker index

    if (key_counter % update_interval == 0) {
      // We don't update the progress bar every time, in extreme cases this may be too slow.
      build_bar.set_progress(key_counter);
    }
    ++key_counter;
  }

  build_bar.mark_as_completed();

  spdlog::info("Built the Python index, releasing the C++ index.");
  merged_chunker_index->clear();
  *merged_chunker_index = ChunkerIndexCpp();
  return py_chunker_index;
}

py::object create_chunker_index(py::object py_table, int num_threads) {
  try {
    spdlog::info("Entered C++ extension for building chunker index.");
    if (num_threads < 0) {
      FAIL(fmt::format("num_threads = {} < 0. Need at least 0 (1) threads.", num_threads));
    }

    num_threads = std::max(num_threads, 1);
    indicators::show_console_cursor(false);  // Hides cursor, just for visuals

    std::vector<ChunkerIndexCpp> thread_chunker_indices =
        calc_thread_chunker_indices(py_table, static_cast<uint32_t>(num_threads));

    ChunkerIndexCpp merged_chunker_index = merge_chunker_indices(&thread_chunker_indices);
    const py::dict result = build_py_chunker_index(&merged_chunker_index);

    indicators::show_console_cursor(true);  // Show cursor again
    spdlog::info("C++ create_chunker_index done, returning.");
    return result;

  } catch (const std::exception& e) {
    const std::string error_msg = fmt::format("Exception occurred in create_chunker_index: {}", e.what());
    spdlog::error(error_msg);
    py::gil_scoped_acquire acquire;
    PyErr_SetString(PyExc_RuntimeError, error_msg.c_str());
    throw py::error_already_set();
  }
}

PYBIND11_MODULE(chunker_extension, m) {
  m.doc() = "Chunker Index Extension Module";
  m.def("create_chunker_index_cpp", &create_chunker_index, py::arg("table"), py::arg("num_threads") = 4);
}