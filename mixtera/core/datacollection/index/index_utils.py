from mixtera.core.datacollection.index.index import ChunkerIndex


def infer_mixture_from_chunkerindex(chunker_index: ChunkerIndex) -> tuple[int, dict[str, float]]:
    """
    Infer the mixture from the result chunker index.
    This is done by calculating the mass of each partition
    and normalizing it to the total mass.

    Returns:
        The inferred mixture
    """
    total_count = 0
    partition_masses: dict[str, int | float] = {}

    def calculate_partition_mass(partition: dict[int, dict[int, list[tuple[int, int]]]]) -> int:
        mass = sum(
            end - start for file_entry in partition.values() for ranges in file_entry.values() for start, end in ranges
        )
        return mass

    for property_combination, partition_entry in chunker_index.items():
        partition_mass = calculate_partition_mass(partition_entry)
        partition_masses[property_combination] = partition_mass
        total_count += partition_mass

    for key in partition_masses:
        partition_masses[key] /= total_count

    return total_count, partition_masses
