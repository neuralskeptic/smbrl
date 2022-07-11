class SetLogging:
    def __init__(self, logger, objects):
        self._logger = logger
        self._objects = objects if isinstance(objects, list) else list(objects)

    def __enter__(self):
        for o in self._objects:
            if hasattr(o, "_logger"):
                o._logger = self._logger
            else:
                pass
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        for o in self._objects:
            if hasattr(o, "_logger"):
                o._logger = None
            else:
                pass
        return self
