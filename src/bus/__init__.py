from collections import defaultdict
from collections.abc import Callable


class Bus:
    def __init__(self):
        self._subs: dict[str, list[Callable]] = defaultdict(list)

    def subscribe(self, topic: str, fn: Callable) -> None:
        self._subs[topic].append(fn)

    def publish(self, topic: str, msg) -> None:
        for fn in self._subs[topic]:
            fn(msg)
