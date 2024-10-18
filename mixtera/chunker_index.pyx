# chunker_index.pyx

# Import necessary modules
import numpy as np
cimport numpy as np
import cython

@cython.boundscheck(False)
@cython.wraparound(False)
def _create_chunker_index_np(
    np.ndarray[np.int32_t, ndim=1] dataset_ids,
    np.ndarray[np.int32_t, ndim=1] file_ids,
    np.ndarray[np.int32_t, ndim=1] interval_starts,
    np.ndarray[np.int32_t, ndim=1] interval_ends,
    np.ndarray[np.int64_t, ndim=1] indicators,
    dict property_arrays,          # Can't specify types here as properties can be mixed types
    list property_columns,
    object MixtureKey_class
):
    """
    Create a chunker index from the given arrays.

    Args:
        dataset_ids: NumPy array of dataset IDs.
        file_ids: NumPy array of file IDs.
        interval_starts: NumPy array of interval starts.
        interval_ends: NumPy array of interval ends.
        indicators: NumPy array of group change indicators.
        property_arrays: Dictionary mapping property names to NumPy arrays.
        property_columns: List of property column names.
        MixtureKey_class: Reference to the MixtureKey class.

    Returns:
        dict: The chunker index.
    """
    cdef Py_ssize_t idx
    cdef Py_ssize_t total_rows = dataset_ids.shape[0]
    cdef dict chunker_index = {}
    cdef object current_mixture_key
    cdef np.int64_t dataset_id, file_id, interval_start, interval_end, indicator
    cdef tuple interval
    cdef dict properties
    cdef object value
    cdef str k

    for idx in range(total_rows):
        dataset_id = dataset_ids[idx]
        file_id = file_ids[idx]
        interval_start = interval_starts[idx]
        interval_end = interval_ends[idx]
        indicator = indicators[idx]
        interval = (interval_start, interval_end)

        # Build properties dictionary per row
        properties = {}
        for k in property_columns:
            value = property_arrays[k][idx]

            # Convert numpy scalars and arrays to Python types
            if isinstance(value, np.generic):
                value = value.item()
            elif isinstance(value, np.ndarray):
                value = value.tolist()

            # Continue with your existing checks
            if value is not None and not (isinstance(value, list) and len(value) == 0):
                if isinstance(value, list):
                    properties[k] = value
                else:
                    properties[k] = [value]

        # Create MixtureKey instance
        current_mixture_key = MixtureKey_class(properties)

        # Append interval to the chunker index
        if current_mixture_key not in chunker_index:
            chunker_index[current_mixture_key] = {}
        if dataset_id not in chunker_index[current_mixture_key]:
            chunker_index[current_mixture_key][dataset_id] = {}
        if file_id not in chunker_index[current_mixture_key][dataset_id]:
            chunker_index[current_mixture_key][dataset_id][file_id] = []
        chunker_index[current_mixture_key][dataset_id][file_id].append(interval)

    return chunker_index
