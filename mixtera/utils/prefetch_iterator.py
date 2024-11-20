from threading import Event, Thread
from typing import Iterator, TypeVar

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
