

class TimeItem:
    def __init__(self):
        self.start_time = 0
        self.end_time = 0
        self.duration = 0

    def init_by_event(self, event):
        self.init(event['ts'], event['dur'])
        return self

    def init(self, start_time, duration):
        self.start_time = start_time
        self.duration = duration
        self.end_time = self.start_time + self.duration

    def get_start_time(self):
        return self.start_time

    def get_end_time(self):
        return self.end_time

    def get_duration(self):
        return self.duration

    def merge(self, t1):
        self.start_time = min(self.start_time, t1.start_time)
        self.end_time = max(self.end_time, t1.end_time)
        self.duration = self.end_time - self.start_time

    def __str__(self):
        return "start_time:" + str(self.start_time) + ", end_time:" + str(self.end_time)


def time_in(time1, time2):
    """
    time1 between time  2
    """
    return approx_gte(time1.get_start_time(), time2.get_start_time()) and approx_lte(time1.get_end_time(),
                                                                                     time2.get_end_time())


def time_before(time1, time2):
    """
    event1 before event 2
    """
    return approx_lte(time1.get_end_time(), time2.get_start_time())


def isclose(x, y, precision=1.e-5):
    return abs(x-y)<=precision

def approx_lte(x, y):
    return x <= y or isclose(x, y)


def approx_gte(x, y):
    return x >= y or isclose(x, y)
