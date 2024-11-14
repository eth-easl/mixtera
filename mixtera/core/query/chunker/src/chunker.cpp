// chunker.cpp
#include "chunker.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <arrow/api.h>
#include <arrow/python/pyarrow.h>
#include <thread>
#include <iostream>
#include <typeinfo>
#include <arrow/util/checked_cast.h>
#include <indicators/progress_bar.hpp>
#include <indicators/block_progress_bar.hpp>
#include <indicators/termcolor.hpp>
#include <indicators/cursor_control.hpp> 


namespace py = pybind11;

bool GetIndexValue(const std::shared_ptr<arrow::Array>& indices, int64_t position, int64_t& out_index) {
    if (position < 0 || position >= indices->length()) {
        std::cerr << "Index position out of bounds: " << position << std::endl;
        return false;
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
            std::cerr << "Unsupported index type: " << indices->type()->ToString() << std::endl;
            return false;
    }
}


void merge_sorted_intervals_inplace(std::vector<Interval>& target_intervals,
                                      std::vector<Interval>& source_intervals) {
      size_t m = target_intervals.size();
      size_t n = source_intervals.size();

      target_intervals.reserve(m + n);

      // Copy source_intervals to target_intervals
      target_intervals.insert(target_intervals.end(),
                              std::make_move_iterator(source_intervals.begin()),
                              std::make_move_iterator(source_intervals.end()));

      // Clear source_intervals to free memory
      source_intervals.clear();
      source_intervals.shrink_to_fit();

      // In-place sort
      std::inplace_merge(target_intervals.begin(), target_intervals.begin() + m, target_intervals.end());
  }


void merge_chunker_indices(std::vector<ChunkerIndexCpp>& thread_indices,
                           ChunkerIndexCpp& merged_index) {
    size_t total_indices = thread_indices.size();
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
        indicators::option::FontStyles{std::vector<indicators::FontStyle>{indicators::FontStyle::bold}}
    };

    std::cout << "Number of per-thread indices: " << total_indices << std::endl;

    for (size_t i = 0; i < total_indices; ++i) {
        size_t num_keys = thread_indices[i].size();
        std::cout << "Thread index " << i << " has " << num_keys << " keys." << std::endl;
    }

    for (auto& local_index : thread_indices) {
        for (auto& [key, datasets] : local_index) {
            // Use merged index's datasets for the given key
            auto& target_datasets = merged_index[key];

            for (auto& [dataset_id, files] : datasets) {
                // Use merged index's files for the given dataset_id
                auto& target_files = target_datasets[dataset_id];

                for (auto& [file_id, intervals] : files) {
                    // Use merged index's intervals for the given file_id
                    auto& target_intervals = target_files[file_id];

                    if (target_intervals.empty()) {
                        target_intervals = std::move(intervals);
                    } else {
                        merge_sorted_intervals_inplace(target_intervals, intervals);
                    }
                }
                // files map's content (intervals) have been moved or cleared
                // We can clear the files map here if desired
                files.clear();  // Safe to clear after iteration
                files = FileIntervals(); // force deallocation by deleting the old object
            }
            // datasets map's content (files) have been cleared
            datasets.clear();  // Safe to clear after iteration
            datasets = DatasetFiles(); // force deallocation
        }
        // local_index's content (datasets) have been cleared
        local_index.clear();  // Clear the local_index to free memory
        local_index = ChunkerIndexCpp(); // force deallocation
        merge_bar.tick();
    }
    // After processing all local indices, we can clear the thread_indices vector itself
    thread_indices.clear();
    thread_indices.shrink_to_fit();
    merge_bar.mark_as_completed();
}

// Helper function to process a range of rows
void process_batch(const std::shared_ptr<arrow::RecordBatch>& batch,
                   const std::vector<std::string>& property_columns,
                   ChunkerIndexCpp& local_chunker_index) {
    try {

        int64_t num_rows = batch->num_rows();

        // Get required columns
        auto dataset_id_array = batch->GetColumnByName("dataset_id");
        auto file_id_array = batch->GetColumnByName("file_id");
        auto interval_start_array = batch->GetColumnByName("interval_start");
        auto interval_end_array = batch->GetColumnByName("interval_end");

        if (!dataset_id_array || !file_id_array || !interval_start_array || !interval_end_array) {
            std::cerr << "One or more required columns are missing in batch." << std::endl;
            return;
        }

        // Cast required arrays
        auto dataset_id_array_int32 = arrow::internal::checked_pointer_cast<arrow::Int32Array>(dataset_id_array);
        auto file_id_array_int32 = arrow::internal::checked_pointer_cast<arrow::Int32Array>(file_id_array);
        auto interval_start_array_int32 = arrow::internal::checked_pointer_cast<arrow::Int32Array>(interval_start_array);
        auto interval_end_array_int32 = arrow::internal::checked_pointer_cast<arrow::Int32Array>(interval_end_array);

        // Prepare property arrays
        std::vector<std::shared_ptr<arrow::Array>> property_arrays;
        for (const auto& col_name : property_columns) {
            auto array = batch->GetColumnByName(col_name);
            if (!array) {
                std::cerr << "Property column '" << col_name << "' not found in batch." << std::endl;
                property_arrays.push_back(nullptr); // Placeholder
                continue;
            }
            property_arrays.push_back(array);
        }

        for (int64_t i = 0; i < num_rows; ++i) {
            // Build the key as a string
            std::string key;

            bool first_prop = true;
            for (size_t j = 0; j < property_columns.size(); ++j) {
                auto array = property_arrays[j];
                std::string col_name = property_columns[j];

                if (!array) {
                    continue;
                }

                if (array->IsNull(i)) {
                    continue;
                }

                if (!first_prop) {
                    key += ";";
                }
                first_prop = false;

                key += col_name + ":";

                auto type_id = array->type_id();

                try {
                    if (type_id == arrow::Type::STRING) {
                        // Handle STRING type
                        auto str_array = arrow::internal::checked_pointer_cast<arrow::StringArray>(array);
                        std::string value = str_array->GetString(i);
                        key += value;
                    } else if (type_id == arrow::Type::LIST) {
                        // Handle LIST type
                        auto list_array = arrow::internal::checked_pointer_cast<arrow::ListArray>(array);
                        if (list_array->IsNull(i)) {
                            continue;
                        }

                        int64_t offset = list_array->value_offset(i);
                        int64_t length = list_array->value_length(i);
                        auto value_array = list_array->values();

                        if (!value_array) {
                            std::cerr << "Value array is null for LIST column '" << col_name << "'" << std::endl;
                            continue;
                        }

                        if (offset < 0 || length < 0 || (offset + length) > value_array->length()) {
                            std::cerr << "Invalid offset/length for LIST column '" << col_name << "' at row " << i << std::endl;
                            continue;
                        }

                        // Build values string
                        std::string values_str;
                        bool first_value = true;

                        if (value_array->type_id() == arrow::Type::STRING) {
                            // LIST of STRING
                            auto str_values = arrow::internal::checked_pointer_cast<arrow::StringArray>(value_array);
                            for (int64_t k = offset; k < offset + length; ++k) {
                                if (str_values->IsNull(k)) {
                                    continue;
                                }
                                std::string val = str_values->GetString(k);
                                if (!first_value) {
                                    values_str += ",";
                                }
                                first_value = false;
                                values_str += val;
                            }
                        } else if (value_array->type_id() == arrow::Type::DICTIONARY) {
                            // LIST of DICTIONARY
                            auto dict_array = arrow::internal::checked_pointer_cast<arrow::DictionaryArray>(value_array);
                            auto dict = dict_array->dictionary();
                            auto dict_values = arrow::internal::checked_pointer_cast<arrow::StringArray>(dict);
                            auto indices = dict_array->indices();

                            if (!dict_values || !indices) {
                                std::cerr << "Dictionary values or indices are null in LIST of DICTIONARY for column '" << col_name << "'" << std::endl;
                                continue;
                            }

                            for (int64_t k = offset; k < offset + length; ++k) {
                                if (dict_array->IsNull(k)) {
                                    continue;
                                }

                                int64_t index = 0;
                                if (!GetIndexValue(indices, k, index)) {
                                    std::cerr << "Failed to get index value at position " << k << " in LIST of DICTIONARY for column '" << col_name << "'" << std::endl;
                                    continue;
                                }

                                if (index < 0 || index >= dict_values->length()) {
                                    std::cerr << "Index out of bounds in dictionary values for LIST of DICTIONARY at position " << k << std::endl;
                                    continue;
                                }

                                std::string val = dict_values->GetString(index);
                                if (!first_value) {
                                    values_str += ",";
                                }
                                first_value = false;
                                values_str += val;
                            }
                        } else {
                            std::cerr << "Unsupported list element type in column '" << col_name << "'. Type: " << value_array->type()->ToString() << std::endl;
                        }

                        key += values_str;

                    } else if (type_id == arrow::Type::DICTIONARY) {
                        // Handle DICTIONARY type
                        auto dict_array = arrow::internal::checked_pointer_cast<arrow::DictionaryArray>(array);

                        if (dict_array->IsNull(i)) {
                            continue;
                        }

                        auto dict = dict_array->dictionary();
                        auto dict_values = arrow::internal::checked_pointer_cast<arrow::StringArray>(dict);
                        auto indices = dict_array->indices();

                        if (!dict_values || !indices) {
                            std::cerr << "Dictionary values or indices are null in DICTIONARY column '" << col_name << "'" << std::endl;
                            continue;
                        }

                        int64_t index = 0;
                        if (!GetIndexValue(indices, i, index)) {
                            std::cerr << "Failed to get index value at position " << i << " in DICTIONARY column '" << col_name << "'" << std::endl;
                            continue;
                        }

                        if (index < 0 || index >= dict_values->length()) {
                            std::cerr << "Index out of bounds in dictionary values for DICTIONARY at position " << i << std::endl;
                            continue;
                        }

                        std::string value = dict_values->GetString(index);
                        key += value;

                    } else {
                        std::cerr << "Unsupported array type in column '" << col_name << "' at row " << i << ". Type: " << array->type()->ToString() << std::endl;
                    }
                } catch (const std::exception& e) {
                    std::cerr << "Exception at row " << i << ", column '" << col_name << "': " << e.what() << std::endl;
                    continue;
                }
            }
            int64_t dataset_id = dataset_id_array_int32->Value(i);
            int64_t file_id = file_id_array_int32->Value(i);
            int64_t interval_start = interval_start_array_int32->Value(i);
            int64_t interval_end = interval_end_array_int32->Value(i);

            if (interval_end < interval_start) {
                std::cerr << "Warning: interval_end = " << interval_end << " < interval_start = " << interval_start << " at row " << i << " (file " << file_id << " dataset " << dataset_id << " ) " << std::endl;
                continue; // Skip invalid intervals
            }


            // Optional: Debugging output
            /*
            if (i < 10) {
                std::cout << "Row " << i << ": dataset_id=" << dataset_id
                          << ", file_id=" << file_id
                          << ", interval_start=" << interval_start
                          << ", interval_end=" << interval_end << std::endl;
            }
            */

            Interval interval = {interval_start, interval_end};

            // Store intervals in the local chunker index
            local_chunker_index[key][dataset_id][file_id].push_back(interval);
        }

        // No need to sort intervals within threads since data is already sorted

    } catch (const std::exception& e) {
        std::cerr << "Exception in process_rows: " << e.what() << std::endl;
        throw;
    }
}
// Main function
py::object create_chunker_index(py::object py_table, int num_threads) {
    try {
        // Prepare per-thread indices
        std::vector<ChunkerIndexCpp> thread_chunker_indices(num_threads);

        {
            // Initialize Arrow C++ and Python bridges
            arrow::py::import_pyarrow();

            // Convert PyArrow Table to C++ Arrow Table
            std::shared_ptr<arrow::Table> table = arrow::py::unwrap_table(py_table.ptr()).ValueOrDie();

            // Identify property columns
            std::vector<std::string> exclude_keys = {"dataset_id", "file_id", "group_id", "interval_start", "interval_end"};
            std::vector<std::string> property_columns;

            for (const auto& field : table->schema()->fields()) {
                std::string col_name = field->name();
                if (std::find(exclude_keys.begin(), exclude_keys.end(), col_name) == exclude_keys.end()) {
                    property_columns.push_back(col_name);
                }
            }

            // Release GIL for multithreading
            py::gil_scoped_release release;

            // Create a TableBatchReader
            arrow::TableBatchReader batch_reader(*table);
            //batch_reader.set_chunksize(1); // Set chunksize (adjust as needed)

            // Read all record batches
            std::vector<std::shared_ptr<arrow::RecordBatch>> batches;
            std::shared_ptr<arrow::RecordBatch> batch;
            while (batch_reader.ReadNext(&batch).ok() && batch) {
                batches.push_back(batch);
            }

            const int64_t total_rows = table->num_rows();
            std::atomic<int64_t> total_rows_processed{0};

            indicators::show_console_cursor(false);
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
                indicators::option::Stream{std::cout}, // Specify the output stream
                indicators::option::FontStyles{std::vector<indicators::FontStyle>{indicators::FontStyle::bold}}
            };

            
            // Determine batch ranges for each thread
            int64_t num_batches = batches.size();
            int64_t batches_per_thread = (num_batches + num_threads - 1) / num_threads;

            // Launch threads
            std::vector<std::thread> threads;
            for (int t = 0; t < num_threads; ++t) {
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
                            std::cerr << "Exception in thread " << t << ": " << e.what() << std::endl;
                            // Handle exception or terminate
                            throw;
                        }
                    });
                }
            }

            // Wait for threads to finish
            for (auto& thread : threads) {
                thread.join();
            }

            std::cout << "All threads finished, clearing batches." << std::endl;
            batches.clear();
            batches.shrink_to_fit();
            table.reset();
        }

        // Merge per-thread chunker indices
        ChunkerIndexCpp merged_chunker_index;

        if (thread_chunker_indices.size() == 1) {
            merged_chunker_index = thread_chunker_indices[0];
        } else {
            std::cout << "Merging per-thread indices" << std::endl;
            merge_chunker_indices(thread_chunker_indices, merged_chunker_index);
        }

        // After merging, we can clear thread_chunker_indices to free memory
        std::cout << "Merged indices, clearing memory." << std::endl;
        thread_chunker_indices.clear();
        thread_chunker_indices.shrink_to_fit();
        // Reacquire GIL before working with Python objects
        std::cout << "Acquiring GIL." << std::endl;
        py::gil_scoped_acquire acquire;
        std::cout << "Acquired GIL." << std::endl;

        // Import the MixtureKey class
        py::object mixture_module = py::module_::import("mixtera.core.query.mixture");
        py::object MixtureKey_class = mixture_module.attr("MixtureKey");

        // Prepare the ChunkerIndex Python dict
        py::dict py_chunker_index;

        size_t total_keys = merged_chunker_index.size();
        const size_t update_interval = std::max<size_t>(std::ceil<size_t>(static_cast<double>(total_keys) * 0.01), static_cast<size_t>(1));
        const size_t max_progress = std::min(update_interval * 100, total_keys);

        // Initialize the building progress bar
        indicators::BlockProgressBar build_bar{
            indicators::option::BarWidth{50},
            indicators::option::Start{"["},
            indicators::option::End{"]"},
            indicators::option::ForegroundColor{indicators::Color::cyan},
            indicators::option::PrefixText{"Building Python object: "},
            indicators::option::ShowElapsedTime{true},
            indicators::option::ShowRemainingTime{true},
            indicators::option::MaxProgress{max_progress},
            indicators::option::Stream{std::cout},
            indicators::option::FontStyles{std::vector<indicators::FontStyle>{indicators::FontStyle::bold}}
        };

        // To minimize memory usage, we will move data when possible
        size_t key_counter = 0;

    for (auto it = merged_chunker_index.begin(); it != merged_chunker_index.end(); ) {
        // Extract the key and datasets by moving them out of the map
        MixtureKeyCpp key = std::move(it->first);
        DatasetFiles datasets = std::move(it->second);

        // Advance iterator before erasing
        auto current_it = it;
        ++it; // Increment iterator before erasing
        merged_chunker_index.erase(current_it);

        std::cout << "Erased from index" << std::endl;

            // Parse the key string back into a properties dictionary
            py::dict py_properties;
            std::string key_str = key;

            // Parse the key string
            // Expected format: "prop1:val1,val2;prop2:val3"

            std::stringstream ss_props(key_str);
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

            std::cout << "Parsed key " << key_str << " into py_properties" << std::endl;
            // Create MixtureKey object
            py::object py_mixture_key = MixtureKey_class(py_properties);
            std::cout << "Created mixturekey object" << std::endl;

        // Build datasets dict
        py::dict py_datasets;
        for (auto& [dataset_id, files] : datasets) {
            py::dict py_files;
            for (auto& [file_id, intervals] : files) {
                // Convert vector of intervals to Python list
                py_files[py::cast(file_id)] = py::cast(std::move(intervals));
                // Clear intervals vector after casting
                intervals.clear();
                intervals.shrink_to_fit();
            }
            // Clear files map to free memory
            files.clear();
            files = FileIntervals(); // Force deallocation (optional)

            py_datasets[py::cast(dataset_id)] = py_files;
        }
        std::cout << "Built dataset dict." << std::endl;

        // Clear datasets map to free memory
        datasets.clear();
        datasets = DatasetFiles(); // Force deallocation (optional)

        // Add to chunker index
        py_chunker_index[py_mixture_key] = py_datasets;

        std::cout << "Added to chunker index." << std::endl;

        // Update progress bar
        if (key_counter % update_interval == 0) {
            std::cout << "Processed " << key_counter << " / " << total_keys << " keys...";
            build_bar.tick();
        }
        ++key_counter;
        }

        build_bar.mark_as_completed();
        indicators::show_console_cursor(true);

        std::cout << "Releasing the C++ index." << std::endl;
        merged_chunker_index.clear();
        std::cout << "Returning from C++" << std::endl;

        // Return the chunker index
        return py_chunker_index;

    } catch (const std::exception& e) {
        std::cerr << "Exception occurred in create_chunker_index: " << e.what() << std::endl;
        throw;
    }
}



PYBIND11_MODULE(chunker_extension, m) {
    m.doc() = "Chunker Index Extension Module";
    m.def("create_chunker_index_cpp", &create_chunker_index, py::arg("table"), py::arg("num_threads") = 4);
}