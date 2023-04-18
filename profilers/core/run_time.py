
class RunTime:
    @property
    def run_time_ms(self):
        raise NotImplementedError

    @property
    def ktime_ns(self):
        return sum(map(lambda k: k.run_time_ns, self.kernels))

    @property
    def kernels(self):
        return []

    @property
    def device(self):
        raise NotImplementedError


class RunTimeMeasurement(RunTime):
    def __init__(self, run_time_ms, device):
        self._run_time_ms = run_time_ms
        self._device = device

    @property
    def run_time_ms(self):
        return self._run_time_ms

    @property
    def kernels(self):
        return self._kernels

    @property
    def device(self):
        return self._device