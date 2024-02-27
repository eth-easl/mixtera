import json
import unittest

from mixtera.core.datacollection.index.parser_collection import MetadataParserFactory, RedPajamaMetadataParser
from mixtera.utils import defaultdict_to_dict


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
        expected = {
            "language": {
                "C": {0: {0: [0, 1]}},  # value with document and list of lines
                "C++": {0: {0: [0, 1]}},
                "CoffeeScript": {0: {0: [0]}},
                "PHP": {0: {0: [1]}},
            },
            "publication_date": {"asd123": {0: {0: [0]}}},
        }

        for line_number, metadata in enumerate(lines):
            red_pajama_metadata_parser.parse(line_number, metadata)

        self.assertEqual(expected, defaultdict_to_dict(red_pajama_metadata_parser._index))

    def test_compress_index(self):
        dataset_id: int = 0
        file_id: int = 0
        red_pajama_metadata_parser = RedPajamaMetadataParser(dataset_id, file_id)

        red_pajama_metadata_parser._index = {
            "language": {
                "C": {0: {0: [0, 2, 4, 9]}},
                "PHP": {0: {0: [1]}},
            },
            "publication_date": {"asd123": {0: {0: [0, 2, 3, 4, 5, 9, 10]}}},
        }

        target_index = {
            "language": {
                "C": {0: {0: [(0, 1), (2, 3), (4, 5), (9, 10)]}},
                "PHP": {0: {0: [(1, 2)]}},
            },
            "publication_date": {"asd123": {0: {0: [(0, 1), (2, 6), (9, 11)]}}},
        }

        red_pajama_metadata_parser._compress_index()
        self.assertEqual(target_index, red_pajama_metadata_parser._index)

    def test_get_index(self):
        dataset_id: int = 0
        file_id: int = 0
        red_pajama_metadata_parser = RedPajamaMetadataParser(dataset_id, file_id)

        red_pajama_metadata_parser._index = {
            "language": {
                "C": {0: {0: [0, 2, 4, 9]}},
                "PHP": {0: {0: [1]}},
            },
            "publication_date": {"asd123": {0: {0: [0, 2, 3, 4, 5, 9, 10]}}},
        }

        target_index = {
            "language": {
                "C": {0: {0: [(0, 1), (2, 3), (4, 5), (9, 10)]}},
                "PHP": {0: {0: [(1, 2)]}},
            },
            "publication_date": {"asd123": {0: {0: [(0, 1), (2, 6), (9, 11)]}}},
        }

        self.assertEqual(target_index, red_pajama_metadata_parser.get_index())


class TestMetadataParserFactory(unittest.TestCase):
    def test_create_metadata_parser(self):
        dataset_id: int = 0
        file_id: int = 0
        metadata_parser = MetadataParserFactory()

        tests_and_targets = [("RED_PAJAMA", RedPajamaMetadataParser)]

        for dtype, ctype in tests_and_targets:
            self.assertIsInstance(metadata_parser.create_metadata_parser(dtype, dataset_id, file_id), ctype)
