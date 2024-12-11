import threading
from typing import Any, Iterator


class TokenizingIterator:
    def __init__(
        self, iterator: Iterator[str], tokenizer: Any, sequence_length: int, batch_size: int, at_least_one_sample: bool
    ) -> None:
        self.iterator = iterator
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.at_least_one_sample = at_least_one_sample
        self.buffer: list[int] = []
        self.chunk_index = 0
        self.eos = False
        self.yielded_samples = 0

    def __iter__(self) -> "TokenizingIterator":
        return self

    def fetch_data(self) -> None:
        texts = []
        for _ in range(self.batch_size):
            try:
                texts.append(next(self.iterator))
            except StopIteration:
                self.eos = True
                break

        if texts:
            encoded_batch = self.tokenizer.batch_encode_plus(
                texts,
                return_attention_mask=False,
                return_token_type_ids=False,
            )
            # Flatten the list of token lists and extends the buffer
            self.buffer.extend([id for ids in encoded_batch["input_ids"] for id in ids])

    def __next__(self) -> list[int]:
        while True:
            current_length = len(self.buffer) - self.chunk_index
            if current_length >= self.sequence_length + 1:
                # Yields a chunk of sequence_length + 1 tokens as per nanotron's clm_process
                start = self.chunk_index
                end = start + self.sequence_length + 1
                chunk = self.buffer[start:end]
                self.chunk_index += self.sequence_length
                self.yielded_samples += 1
                return chunk

            if not self.eos:
                self.fetch_data()
                continue

            # End of data, check if we have yielded at least one sample
            if self.yielded_samples == 0 and current_length > 0 and self.at_least_one_sample:
                # Pad remaining tokens by repeating them to reach the desired length
                needed_tokens = self.sequence_length + 1 - current_length
                repeats = (needed_tokens + current_length - 1) // current_length
                padded_tokens = (self.buffer[self.chunk_index :] * (1 + repeats))[: self.sequence_length + 1]
                chunk = padded_tokens
                self.yielded_samples += 1
                return chunk

            raise StopIteration


class ThreadedTokenizingIterator:
    def __init__(
        self, iterator: Iterator[str], tokenizer: Any, sequence_length: int, batch_size: int, at_least_one_sample: bool
    ) -> None:
        self.iterator = iterator
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.at_least_one_sample = at_least_one_sample

        self.buffer: list[int] = []
        self.chunk_index = 0
        self.eos = False
        self.yielded_samples = 0

        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
        self.background_thread = threading.Thread(target=self.prefetch_loop)
        self.background_thread.start()

    def __iter__(self) -> "ThreadedTokenizingIterator":
        return self

    def prefetch_loop(self) -> None:
        while not self.eos:
            texts = []
            local_eos = False
            for _ in range(self.batch_size):
                try:
                    texts.append(next(self.iterator))
                except StopIteration:
                    local_eos = True

            if texts:
                encoded_batch = self.tokenizer.batch_encode_plus(
                    texts,
                    return_attention_mask=False,
                    return_token_type_ids=False,
                )
                with self.lock:
                    self.buffer.extend([id for ids in encoded_batch["input_ids"] for id in ids])
                    if local_eos:
                        self.eos = True
                    self.condition.notify_all()
            else:
                # No more texts to process
                with self.lock:
                    self.eos = True
                    self.condition.notify_all()

    def __next__(self) -> list[int]:
        while True:
            with self.lock:
                current_length = len(self.buffer) - self.chunk_index
                if current_length >= self.sequence_length + 1:
                    start = self.chunk_index
                    end = start + self.sequence_length + 1
                    chunk = self.buffer[start:end]
                    self.chunk_index += self.sequence_length
                    self.yielded_samples += 1
                    return chunk

                if self.eos:
                    if self.yielded_samples == 0 and current_length > 0 and self.at_least_one_sample:
                        # Pad the remaining tokens to reach the desired length
                        needed_tokens = self.sequence_length + 1 - current_length
                        repeats = (needed_tokens + current_length - 1) // current_length
                        padded_tokens = (self.buffer[self.chunk_index :] * (1 + repeats))[: self.sequence_length + 1]
                        chunk = padded_tokens
                        self.yielded_samples += 1
                        return chunk

                    raise StopIteration

                # Not enough tokens and data is not over, wait for more data
                self.condition.wait()
