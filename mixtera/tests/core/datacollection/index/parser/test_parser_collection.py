import json
import unittest
from typing import Any, Optional

from mixtera.core.datacollection.index.parser.metadata_parser import MetadataParser
from mixtera.core.datacollection.index.parser.parser_collection import (
    MetadataParserFactory,
    RedPajamaMetadataParser,
    SlimPajamaMetadataParser,
)


class TestRedPajamaMetadataParser(unittest.TestCase):
    def test_parse(self):
        dataset_id: int = 0
        file_id: int = 0
        red_pajama_metadata_parser = RedPajamaMetadataParser(dataset_id, file_id)

        elem1 = json.loads(
            """{
       "text":"...",
       "meta":{
          "publication_date": "asd123",
          "content_hash":"bab3317c67f40063ff7a69f3bcc74bb0",
          "timestamp":"",
          "source":"github",
          "line_count":64,
          "max_line_length":95,
          "avg_line_length":32.140625,
          "alnum_prop":0.6684491978609626,
          "repo_name":"JoshEngebretson/duktape",
          "id":"29411796cbb56b7d91920771a24db254493ccfc8",
          "size":"2057",
          "binary":false,
          "copies":"1",
          "ref":"refs/heads/master",
          "path":"src/duk_heap_hashstring.c",
          "mode":"33188",
          "license":"mit",
          "language":[
             {
                "name":"C",
                "bytes":"1972812"
             },
             {
                "name":"C++",
                "bytes":"20922"
             },
             {
                "name":"CoffeeScript",
                "bytes":"895"
             }
          ]
       }
    }"""
        )

        elem2 = json.loads(
            """{
       "text":"...",
       "meta":{
          "content_hash":"009fcdd5cf234bb851939760b7bb2bec",
          "timestamp":"",
          "source":"github",
          "line_count":30,
          "max_line_length":147,
          "avg_line_length":36.93333333333333,
          "alnum_prop":0.6796028880866426,
          "repo_name":"lzpfmh/runkit",
          "id":"44237969acec682b89517ba99d074aa575d5c09a",
          "size":"1108",
          "binary":false,
          "copies":"4",
          "ref":"refs/heads/master",
          "path":"tests/Runkit_Sandbox_Parent__.echo.access.phpt",
          "mode":"33188",
          "license":"bsd-3-clause",
          "language":[
             {
                "name":"C",
                "bytes":"263129"
             },
             {
                "name":"C++",
                "bytes":"372"
             },
             {
                "name":"PHP",
                "bytes":"141611"
             }
          ]
       }
    }"""
        )

        elem3 = json.loads(
            """{
       "text":"...",
       "meta":{}
    }"""
        )

        elem4 = json.loads(
            """{
       "text":"..."
    }"""
        )

        lines = [elem1, elem2, elem3, elem4]
        expected = [
            {
                "sample_id": 0,
                "language": ["C", "C++", "CoffeeScript"],
                "publication_date": "asd123",
            },
            {
                "sample_id": 1,
                "language": ["C", "C++", "PHP"],
                "publication_date": None,  # Publication date is missing
            },
            {
                "sample_id": 2,
                "language": None,
                "publication_date": None,  # No meta data
            },
            {
                "sample_id": 3,
                "language": None,
                "publication_date": None,  # No meta field at all
            },
        ]

        for line_number, metadata in enumerate(lines):
            red_pajama_metadata_parser.parse(line_number, metadata)

        self.assertEqual(expected, red_pajama_metadata_parser.metadata)

    def test_parse_empty_meta(self):
        dataset_id: int = 0
        file_id: int = 0
        red_pajama_metadata_parser = RedPajamaMetadataParser(dataset_id, file_id)

        elem_empty_meta = json.loads(
            """{
       "text":"...",
       "meta":{}
    }"""
        )

        red_pajama_metadata_parser.parse(0, elem_empty_meta)

        expected = [
            {
                "sample_id": 0,
                "language": None,
                "publication_date": None,
            }
        ]

        self.assertEqual(expected, red_pajama_metadata_parser.metadata)

    def test_parse_no_meta(self):
        dataset_id: int = 0
        file_id: int = 0
        red_pajama_metadata_parser = RedPajamaMetadataParser(dataset_id, file_id)

        elem_no_meta = json.loads(
            """{
       "text":"..."
    }"""
        )

        red_pajama_metadata_parser.parse(0, elem_no_meta)

        expected = [
            {
                "sample_id": 0,
                "language": None,
                "publication_date": None,
            }
        ]

        self.assertEqual(expected, red_pajama_metadata_parser.metadata)


class TestSlimPajamaMetadataParser(unittest.TestCase):
    def test_parse(self):
        dataset_id: int = 0
        file_id: int = 0
        slim_pajama_metadata_parser = SlimPajamaMetadataParser(dataset_id, file_id)

        elem = json.loads(
            """{
       "text":"...",
       "meta":{
          "redpajama_set_name": "RedPajamaCommonCrawl"
       }
    }"""
        )

        expected = [
            {
                "sample_id": 0,
                "redpajama_set_name": "RedPajamaCommonCrawl",
            }
        ]

        slim_pajama_metadata_parser.parse(0, elem)
        self.assertEqual(expected, slim_pajama_metadata_parser.metadata)

    def test_parse_no_meta(self):
        dataset_id: int = 0
        file_id: int = 0
        slim_pajama_metadata_parser = SlimPajamaMetadataParser(dataset_id, file_id)

        elem_no_meta = json.loads(
            """{
       "text":"..."
    }"""
        )

        with self.assertRaises(RuntimeError):
            slim_pajama_metadata_parser.parse(0, elem_no_meta)


class TestMetadataParserFactory(unittest.TestCase):
    def test_create_metadata_parser(self):
        dataset_id: int = 0
        file_id: int = 0
        metadata_parser_factory = MetadataParserFactory()

        tests_and_targets = [("RED_PAJAMA", RedPajamaMetadataParser), ("SLIM_PAJAMA", SlimPajamaMetadataParser)]

        for dtype, ctype in tests_and_targets:
            parser = metadata_parser_factory.create_metadata_parser(dtype, dataset_id, file_id)
            self.assertIsInstance(parser, ctype)
            self.assertEqual(parser.dataset_id, dataset_id)
            self.assertEqual(parser.file_id, file_id)

    def test_add_and_remove_parser(self):
        metadata_parser_factory = MetadataParserFactory()

        class DummyParser(MetadataParser):
            @classmethod
            def get_properties(cls) -> list:
                return []

            def parse(self, line_number: int, payload: Any, **kwargs: Optional[dict[Any, Any]]) -> None:
                pass

        # Test adding a new parser
        self.assertTrue(metadata_parser_factory.add_parser("DUMMY", DummyParser))

        # Test adding an existing parser without overwrite
        self.assertFalse(metadata_parser_factory.add_parser("DUMMY", DummyParser))

        # Test adding an existing parser with overwrite
        self.assertTrue(metadata_parser_factory.add_parser("DUMMY", DummyParser, overwrite=True))

        # Test removing an existing parser
        metadata_parser_factory.remove_parser("DUMMY")
        with self.assertRaises(ModuleNotFoundError):
            metadata_parser_factory.create_metadata_parser("DUMMY", 0, 0)

        # Test removing a non-existing parser (should not raise an error)
        metadata_parser_factory.remove_parser("NON_EXISTING")

    def test_create_nonexistent_parser(self):
        metadata_parser_factory = MetadataParserFactory()

        with self.assertRaises(ModuleNotFoundError):
            metadata_parser_factory.create_metadata_parser("NON_EXISTING", 0, 0)
