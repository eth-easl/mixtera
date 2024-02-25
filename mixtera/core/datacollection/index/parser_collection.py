from enum import Enum
from typing import Any, Optional

from mixtera.core.datacollection import IndexType
from mixtera.core.datacollection.index import MetadataParser
from mixtera.utils import ranges


class RedPajamaMetadataParser(MetadataParser):
  """
  Metadata parser class for the RedPajama dataset.
  """
  target_index_fields = ["language", "publication_date"]

  def __init__(self, dataset_id: int, file_id: int):
    super().__init__(dataset_id, file_id)

  def parse(self, line_number: int, metadata: Any,
            **kwargs: Optional[dict[Any, Any]]) -> None:
    for index_field in RedPajamaMetadataParser.target_index_fields:  # pylint: disable=consider-using-dict-items
      if index_field not in metadata:
        continue
      value = metadata[index_field]

      if index_field == "language":
        for language in value:
          lang_name = language["name"]
          self._index[index_field][lang_name][self.dataset_id][
            self.file_id].append(line_number)
      else:
        # TODO(#11): Support numerical buckets, not just categorical
        # logger.info(f"for index {index_field} the value is {value}")
        self._index[index_field][value][self.dataset_id][
          self.file_id].append(line_number)

  def _compress_index(self):
    """
    Compresses the internal index, reducing contiguous line ranges to spans.
    E.g. [1,2,3,5,6] --> [(1,3), (5,6)]. All modifications are done in place
    on the index.
    """
    for index_field, buckets in self._index.items():
      for _, bucket_vals in buckets.items():
        bucket_vals[self.dataset_id][self.file_id] = ranges(
          bucket_vals[self.dataset_id][self.file_id])

  def get_index(self) -> IndexType:
    self._compress_index()
    return self._index


class MetadataParserRegistry(Enum):
  """Each metadata parser is registered with this enum."""
  RED_PAJAMA = RedPajamaMetadataParser


class MetadataParserFactory:
  """Handles the creation of metadata parsers."""
  @staticmethod
  def create_metadata_parser(parser_type: MetadataParserRegistry,
                             dataset_id: int, file_id: int) -> MetadataParser:
    if parser_type == MetadataParserRegistry.RED_PAJAMA:
      return RedPajamaMetadataParser(dataset_id, file_id)
    raise NotImplementedError(f"The {parser_type} metadata parser is not "
                              "implemented!")
