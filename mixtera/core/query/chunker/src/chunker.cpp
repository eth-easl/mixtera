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



std::vector<Interval> merge_sorted_intervals(const std::vector<Interval>& list1, const std::vector<Interval>& list2) {
    std::vector<Interval> merged;
    merged.reserve(list1.size() + list2.size());
    size_t i = 0, j = 0;

    while (i < list1.size() && j < list2.size()) {
        if (list1[i] <= list2[j]) {
            merged.push_back(list1[i++]);
        } else {
            merged.push_back(list2[j++]);
        }
    }
    while (i < list1.size()) {
        merged.push_back(list1[i++]);
    }
    while (j < list2.size()) {
        merged.push_back(list2[j++]);
    }
    return merged;
}

// Function to merge per-thread chunker indices
void merge_chunker_indices(const std::vector<ChunkerIndexCpp>& thread_indices,
                           ChunkerIndexCpp& merged_index) {
    for (const auto& local_index : thread_indices) {
        for (const auto& [key, datasets] : local_index) {
            auto& target_datasets = merged_index[key];
            for (const auto& [dataset_id, files] : datasets) {
                auto& target_files = target_datasets[dataset_id];
                for (const auto& [file_id, intervals] : files) {
                    auto& target_intervals = target_files[file_id];
                    if (target_intervals.empty()) {
                        // If no intervals yet, assign directly
                        target_intervals = intervals;
                    } else {
                        // Merge sorted intervals
                        target_intervals = merge_sorted_intervals(target_intervals, intervals);
                    }
                }
            }
        }
    }
}

// Helper function to process a range of rows
void process_rows(const std::shared_ptr<arrow::Table>& table,
                  int64_t start_row,
                  int64_t end_row,
                  const std::vector<std::string>& property_columns,
                  ChunkerIndexCpp& local_chunker_index) {
    try {

        if (start_row < 0 || end_row > table->num_rows() || start_row >= end_row) {
            std::cerr << "Invalid row indices: start_row=" << start_row << ", end_row=" << end_row << std::endl;
            return;
        }

        // Get columns and check that they exist
        auto dataset_id_column = table->GetColumnByName("dataset_id");
        auto file_id_column = table->GetColumnByName("file_id");
        auto interval_start_column = table->GetColumnByName("interval_start");
        auto interval_end_column = table->GetColumnByName("interval_end");

        if (!dataset_id_column || !file_id_column || !interval_start_column || !interval_end_column) {
            std::cerr << "One or more required columns ('dataset_id', 'file_id', 'interval_start', 'interval_end') are missing." << std::endl;
            return;
        }

        // Since we've called CombineChunks, each column should have only one chunk
        if (dataset_id_column->num_chunks() != 1 || file_id_column->num_chunks() != 1 ||
            interval_start_column->num_chunks() != 1 || interval_end_column->num_chunks() != 1) {
            std::cerr << "Columns have unexpected number of chunks after CombineChunks." << std::endl;
            return;
        }

        std::shared_ptr<arrow::Array> dataset_id_array = dataset_id_column->chunk(0);
        std::shared_ptr<arrow::Array> file_id_array = file_id_column->chunk(0);
        std::shared_ptr<arrow::Array> interval_start_array = interval_start_column->chunk(0);
        std::shared_ptr<arrow::Array> interval_end_array = interval_end_column->chunk(0);

        // **Print out the data types for debugging**
        std::cout << "Dataset ID column type: " << dataset_id_array->type()->ToString()
                    << ", array class: " << typeid(*dataset_id_array.get()).name() << std::endl;

        std::cout << "File ID column type: " << file_id_array->type()->ToString()
                    << ", array class: " << typeid(*file_id_array.get()).name() << std::endl;

        std::cout << "Interval Start column type: " << interval_start_array->type()->ToString()
                    << ", array class: " << typeid(*interval_start_array.get()).name() << std::endl;

        std::cout << "Interval End column type: " << interval_end_array->type()->ToString()
                    << ", array class: " << typeid(*interval_end_array.get()).name() << std::endl;

        // Cast arrays to appropriate types and check
        auto dataset_id_array_int32 = arrow::internal::checked_pointer_cast<arrow::Int32Array>(dataset_id_array);
        auto file_id_array_int32 = arrow::internal::checked_pointer_cast<arrow::Int32Array>(file_id_array);
        auto interval_start_array_int32 = arrow::internal::checked_pointer_cast<arrow::Int32Array>(interval_start_array);
        auto interval_end_array_int32 = arrow::internal::checked_pointer_cast<arrow::Int32Array>(interval_end_array);

        if (!dataset_id_array_int32 || !file_id_array_int32 || !interval_start_array_int32 || !interval_end_array_int32) {
            std::cerr << "Failed to cast arrays to Int32Array." << std::endl;
            return;
        }

        // Check that arrays have sufficient length
        int64_t array_length = dataset_id_array_int32->length();
        if (array_length < end_row) {
            std::cerr << "Array length is less than end_row: array_length=" << array_length << ", end_row=" << end_row << std::endl;
            return;
        }

        // Prepare property arrays
        std::vector<std::shared_ptr<arrow::Array>> property_arrays;
        for (const auto& col_name : property_columns) {
            auto column = table->GetColumnByName(col_name);
            if (!column) {
                std::cerr << "Property column '" << col_name << "' not found in table." << std::endl;
                property_arrays.push_back(nullptr); // Placeholder to maintain alignment
                continue;
            }

            if (column->num_chunks() != 1) {
                std::cerr << "Property column '" << col_name << "' has unexpected number of chunks after CombineChunks." << std::endl;
                property_arrays.push_back(nullptr);
                continue;
            }

            property_arrays.push_back(column->chunk(0));
        }


        for (int64_t i = start_row; i < end_row; ++i) {
            MixtureKeyCpp key;

            for (size_t j = 0; j < property_columns.size(); ++j) {
                auto array = property_arrays[j];
                std::string col_name = property_columns[j];

                if (!array) {
                    continue;
                }

                if (i >= array->length()) {
                    std::cerr << "Index out of bounds for property '" << col_name << "' at row " << i << std::endl;
                    continue;
                }

                if (array->IsNull(i)) {
                    continue;
                }


                auto type_id = array->type_id();

                try {
                    if (type_id == arrow::Type::STRING) {
                        // Handle STRING type
                        auto str_array = arrow::internal::checked_pointer_cast<arrow::StringArray>(array);
                        std::string value = str_array->GetString(i);
                        key.properties[col_name] = {value};
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

                        std::vector<std::string> values;

                        if (value_array->type_id() == arrow::Type::STRING) {
                            // LIST of STRING
                            auto str_values = arrow::internal::checked_pointer_cast<arrow::StringArray>(value_array);
                            for (int64_t k = offset; k < offset + length; ++k) {
                                if (str_values->IsNull(k)) {
                                    continue;
                                }
                                std::string val = str_values->GetString(k);
                                values.push_back(val);
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
                                values.push_back(val);
                            }
                        } else {
                            std::cerr << "Unsupported list element type in column '" << col_name << "'. Type: " << value_array->type()->ToString() << std::endl;
                        }
                        key.properties[col_name] = values;
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
                        key.properties[col_name] = {value};
                    } else {
                        std::cerr << "Unsupported array type in column '" << col_name << "' at row " << i << ". Type: " << array->type()->ToString() << std::endl;
                    }
                } catch (const std::exception& e) {
                    std::cerr << "Exception at row " << i << ", column '" << col_name << "': " << e.what() << std::endl;
                    continue;
                }
            }

            // Access dataset_id, file_id, interval_start, and interval_end
            if (i >= dataset_id_array_int32->length() || i >= file_id_array_int32->length() ||
                i >= interval_start_array_int32->length() || i >= interval_end_array_int32->length()) {
                std::cerr << "Index out of bounds at row " << i << std::endl;
                continue;
            }

            int64_t dataset_id = static_cast<int64_t>(dataset_id_array_int32->Value(i));
            int64_t file_id = static_cast<int64_t>(file_id_array_int32->Value(i));
            int64_t interval_start = static_cast<int64_t>(interval_start_array_int32->Value(i));
            int64_t interval_end = static_cast<int64_t>(interval_end_array_int32->Value(i));

            if (interval_end < interval_start) {
                std::cerr << "Warning: interval_end = " << interval_end << " < interval_start = " << interval_start << " at row " << i << " (file " << file_id << " dataset " << dataset_id << " ) " << std::endl;
                continue; // Skip invalid intervals
            }


            // Optional: Debugging output
            
            if (i < 10) {
                std::cout << "Row " << i << ": dataset_id=" << dataset_id
                          << ", file_id=" << file_id
                          << ", interval_start=" << interval_start
                          << ", interval_end=" << interval_end << std::endl;
            }
            

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
        // Initialize Arrow C++ and Python bridges
        arrow::py::import_pyarrow();

        // Convert PyArrow Table to C++ Arrow Table
        std::shared_ptr<arrow::Table> table = arrow::py::unwrap_table(py_table.ptr()).ValueOrDie();

        std::shared_ptr<arrow::Table> combined_table;
        arrow::Result<std::shared_ptr<arrow::Table>> combine_result = table->CombineChunks();
        if (!combine_result.ok()) {
            std::cerr << "Error combining chunks: " << combine_result.status().message() << std::endl;
            throw std::runtime_error("Failed to combine chunks");
        } else {
            combined_table = combine_result.ValueOrDie();
        }
        std::cout << "Combined table." << std::endl;

        int64_t num_rows = combined_table->num_rows();

        // Identify property columns
        std::vector<std::string> exclude_keys = {"dataset_id", "file_id", "group_id", "interval_start", "interval_end"};
        std::vector<std::string> property_columns;

        for (const auto& field : combined_table->schema()->fields()) {
            std::string col_name = field->name();
            if (std::find(exclude_keys.begin(), exclude_keys.end(), col_name) == exclude_keys.end()) {
                property_columns.push_back(col_name);
            }
        }

        // Release GIL for multithreading
        py::gil_scoped_release release;

        // Determine row ranges for each thread
        int64_t rows_per_thread = (num_rows + num_threads - 1) / num_threads;

        // Prepare per-thread indices
        std::vector<ChunkerIndexCpp> thread_chunker_indices(num_threads);

        // Launch threads
        std::vector<std::thread> threads;
        for (int t = 0; t < num_threads; ++t) {
            int64_t start_row = t * rows_per_thread;
            int64_t end_row = std::min(start_row + rows_per_thread, num_rows);
            if (start_row < end_row) {
                threads.emplace_back([&, start_row, end_row, t]() {
                    try {
                        // Each thread processes its assigned rows
                        process_rows(combined_table, start_row, end_row, property_columns, thread_chunker_indices[t]);
                    } catch (const std::exception& e) {
                        std::cerr << "Exception in thread " << t << ": " << e.what() << std::endl;
                        // Handle exception or terminate
                        std::terminate();
                    }
                });
            }
        }

        // Wait for threads to finish
        for (auto& thread : threads) {
            thread.join();
        }

        // Merge per-thread chunker indices
        ChunkerIndexCpp merged_chunker_index;
        std::cout << "Merging per-thread indices" << std::endl;
        merge_chunker_indices(thread_chunker_indices, merged_chunker_index);

        // Reacquire GIL before working with Python objects
        py::gil_scoped_acquire acquire;
        std::cout << "Acquired GIL." << std::endl;

        // Import the MixtureKey class
        py::object mixture_module = py::module_::import("mixtera.core.query.mixture");
        py::object MixtureKey_class = mixture_module.attr("MixtureKey");

        // Prepare the ChunkerIndex Python dict
        py::dict py_chunker_index;

        for (const auto& [key, datasets] : merged_chunker_index) {
            // Convert MixtureKeyCpp to MixtureKey Python object
            py::dict py_properties;
            for (const auto& [prop_name, prop_values] : key.properties) {
                py::list py_values;
                for (const auto& val : prop_values) {
                    py_values.append(py::str(val));
                }
                py_properties[py::str(prop_name)] = py_values;
            }
            // Create MixtureKey object
            py::object py_mixture_key = MixtureKey_class(py_properties);

            // Build datasets dict
            py::dict py_datasets;
            for (const auto& [dataset_id, files] : datasets) {
                py::dict py_files;
                for (const auto& [file_id, intervals] : files) {
                    py::list py_intervals;
                    for (const auto& interval : intervals) {
                        py_intervals.append(py::make_tuple(interval.first, interval.second));
                    }
                    py_files[py::int_(file_id)] = py_intervals;
                }
                py_datasets[py::int_(dataset_id)] = py_files;
            }

            // Add to chunker index
            py_chunker_index[py_mixture_key] = py_datasets;
        }

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