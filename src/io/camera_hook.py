import traceback


class Hooked:
    """Runs an optional process(image, t) per frame and keeps its newest result."""

    def __init__(self, process):
        self._process = process
        self._result = None
        self._warned = False

    def _run(self, image, t):
        if self._process is None:
            return None
        try:
            return self._process(image, t)
        except Exception:
            # a broken hook must not stop the frames; say why once and keep going
            if not self._warned:
                self._warned = True
                traceback.print_exc()
            return None

    def latest(self):
        return self._result
