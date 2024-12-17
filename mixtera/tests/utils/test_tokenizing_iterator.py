import unittest
from unittest.mock import MagicMock

from mixtera.utils.tokenizing_iterator import ThreadedTokenizingIterator, TokenizingIterator


class TestTokenizingIterators(unittest.TestCase):
    def setUp(self):
        self.iterator_classes = [TokenizingIterator, ThreadedTokenizingIterator]

    def test_yields_correct_chunks_with_enough_data(self):
        """Test iterators yield correct chunks when there is enough data."""
        for iterator_class in self.iterator_classes:
            with self.subTest(iterator_class=iterator_class):
                texts = ["Hello world", "This is a test", "More data here"]
                iterator = iter(texts)

                # Mock tokenizer
                tokenizer = MagicMock()
                tokenizer.batch_encode_plus.side_effect = lambda texts, **kwargs: {
                    "input_ids": [[ord(char) for char in text] for text in texts]
                }

                sequence_length = 5
                batch_size = 2
                at_least_one_sample = False

                tokenizing_iterator = iterator_class(
                    iterator, tokenizer, sequence_length, batch_size, at_least_one_sample
                )

                outputs = []
                try:
                    while True:
                        chunk = next(tokenizing_iterator)
                        outputs.append(chunk)
                except StopIteration:
                    pass
                finally:
                    # Ensure background thread is cleaned up
                    if isinstance(tokenizing_iterator, ThreadedTokenizingIterator):
                        tokenizing_iterator.background_thread.join()

                # Expected tokens
                tokens = [ord(char) for text in texts for char in text]
                expected_chunks = []
                chunk_size = sequence_length + 1
                for i in range(0, len(tokens) - chunk_size + 1, sequence_length):
                    expected_chunks.append(tokens[i : i + chunk_size])

                self.assertEqual(outputs, expected_chunks)

    def test_handles_end_of_data_with_padding(self):
        """Test iterators handle end of data correctly with padding when at_least_one_sample is True."""
        for iterator_class in self.iterator_classes:
            with self.subTest(iterator_class=iterator_class):
                texts = ["Hi"]
                iterator = iter(texts)

                tokenizer = MagicMock()
                tokenizer.batch_encode_plus.side_effect = lambda texts, **kwargs: {
                    "input_ids": [[ord(char) for char in text] for text in texts]
                }

                sequence_length = 5
                batch_size = 2
                at_least_one_sample = True

                tokenizing_iterator = iterator_class(
                    iterator, tokenizer, sequence_length, batch_size, at_least_one_sample
                )

                # Collect outputs
                outputs = []
                try:
                    while True:
                        chunk = next(tokenizing_iterator)
                        outputs.append(chunk)
                except StopIteration:
                    pass
                finally:
                    if isinstance(tokenizing_iterator, ThreadedTokenizingIterator):
                        tokenizing_iterator.background_thread.join()

                # Expected padded tokens
                tokens = [ord("H"), ord("i")]
                current_length = len(tokens)
                needed_tokens = sequence_length + 1 - current_length
                repeats = (needed_tokens + current_length - 1) // current_length
                padded_tokens = (tokens * (1 + repeats))[: sequence_length + 1]
                expected_chunks = [padded_tokens]

                self.assertEqual(outputs, expected_chunks)

    def test_raises_stop_iteration_when_no_data(self):
        """Test iterators raise StopIteration when there's not enough data and at_least_one_sample is False."""
        for iterator_class in self.iterator_classes:
            with self.subTest(iterator_class=iterator_class):
                texts = ["Hi"]
                iterator = iter(texts)

                tokenizer = MagicMock()
                tokenizer.batch_encode_plus.side_effect = lambda texts, **kwargs: {
                    "input_ids": [[ord(char) for char in text] for text in texts]
                }

                sequence_length = 5
                batch_size = 2
                at_least_one_sample = False

                tokenizing_iterator = iterator_class(
                    iterator, tokenizer, sequence_length, batch_size, at_least_one_sample
                )

                try:
                    next(tokenizing_iterator)
                    self.fail("Expected StopIteration exception not raised")
                except StopIteration:
                    pass
                finally:
                    if isinstance(tokenizing_iterator, ThreadedTokenizingIterator):
                        tokenizing_iterator.background_thread.join()

    def test_yields_sample_when_at_least_one_sample_false_with_sufficient_data(self):
        """Test iterators yield chunks correctly when at_least_one_sample is False and there is sufficient data."""
        for iterator_class in self.iterator_classes:
            with self.subTest(iterator_class=iterator_class):
                texts = ["Hello world"]
                iterator = iter(texts)

                tokenizer = MagicMock()
                tokenizer.batch_encode_plus.side_effect = lambda texts, **kwargs: {
                    "input_ids": [[ord(char) for char in text] for text in texts]
                }

                sequence_length = 5
                batch_size = 2
                at_least_one_sample = False

                tokenizing_iterator = iterator_class(
                    iterator, tokenizer, sequence_length, batch_size, at_least_one_sample
                )

                outputs = []
                try:
                    while True:
                        chunk = next(tokenizing_iterator)
                        outputs.append(chunk)
                except StopIteration:
                    pass
                finally:
                    if isinstance(tokenizing_iterator, ThreadedTokenizingIterator):
                        tokenizing_iterator.background_thread.join()

                # Expected tokens
                tokens = [ord(char) for text in texts for char in text]
                chunk_size = sequence_length + 1
                expected_chunks = []
                for i in range(0, len(tokens) - chunk_size + 1, sequence_length):
                    expected_chunks.append(tokens[i : i + chunk_size])

                self.assertEqual(outputs, expected_chunks)
