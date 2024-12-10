import queue
from threading import Event, Thread
from typing import Generic, Iterator, TypeVar

T = TypeVar("T")


class PrefetchFirstItemIterator(Iterator[T]):
    def __init__(self, iterator: Iterator[T]):
        self.iterator = iterator
        self.prefetch_event = Event()
        self.first_item_consumed = False
        self.first_item: T | None = None
        self.prefetch_thread = Thread(target=self._prefetch)
        self.prefetch_thread.start()

    def _prefetch(self) -> None:
        try:
            self.first_item = next(self.iterator)
        except StopIteration:
            self.first_item = None
        finally:
            self.prefetch_event.set()

    def __next__(self) -> T:
        if not self.first_item_consumed:
            self.prefetch_event.wait()
            self.first_item_consumed = True
            if self.first_item is not None:
                return self.first_item

            raise StopIteration()

        return next(self.iterator)


class PrefetchOneItemIterator(Iterator[T], Generic[T]):
    def __init__(self, iterator: Iterator[T]):
        self.iterator = iterator
        # Use a queue with maxsize=1 to hold the prefetched item
        self.queue: "queue.Queue[T | None]" = queue.Queue(maxsize=1)
        self.prefetch_thread = Thread(target=self._prefetch)
        self.prefetch_thread.daemon = True  # Daemonize thread to exit when the main program does
        self.prefetch_thread.start()

    def _prefetch(self) -> None:
        try:
            for item in self.iterator:
                # Put items into the queue, blocking if necessary until a free slot is available
                self.queue.put(item)
            # Signal that the iterator is exhausted by putting a sentinel value (None)
            self.queue.put(None)
        except Exception as e:
            # If an exception occurs, put it into the queue to be raised in the main thread
            self.queue.put(e)

    def __next__(self) -> T:
        item = self.queue.get()
        if isinstance(item, Exception):
            # Re-raise any exceptions from the prefetch thread
            raise item
        if item is None:
            # Iterator is exhausted
            raise StopIteration
        return item

    def __iter__(self) -> "PrefetchOneItemIterator[T]":
        return self
