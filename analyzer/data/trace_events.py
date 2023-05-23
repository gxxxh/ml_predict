import logging

EVENT_TYPE_METADATA = 'M'
EVENT_TYPE_COMPLETE = 'X'

THREAD_TYPE_STEPS = "Steps"
THREAD_TYPE_SCOPE = "TensorFlow Name Scoep"
THREAD_TYPE_OPS = "TensorFolw Ops"
SCOPE_TYPE_FORWARD = "sequencial"
SCOPE_TYPE_LOSS = "loss"
SCOPE_TYPE_BACKWARD = "gradient_tape"


class Process:
    def __init__(self, event):
        """
        device:GPU or host CPU
        """
        self.pid = event["pid"]
        self.name = event["args"]["name"]
        self.threads = {}
        self.memcpyH2D_thread = None
        self.memcpyD2H_thread = None
        self.steps_thread = None
        self.scope_thread = None
        self.ops_thread = None

    def check_event(self, event):
        if (len(event.keys())) == 0:
            return False
        pid = event['pid']
        if pid != self.pid:
            return False
        return True

    def add_meta_event(self, event):
        if self.check_event(event) == False:
            return
        threadid = event['tid']
        if threadid not in self.threads.keys():
            thread = Thread(event)
            self.threads[thread.tid] = thread

    def add_type_event(self, event):
        if self.check_event(event) == False:
            return
        threadid = event['tid']
        if threadid not in self.threads.keys():
            logging.WARN(f'no thread for event: {event}')
        self.threads[threadid].add_event(event)

    def get_thread_by_id(self, tid):
        if tid not in self.threads.keys():
            return None
        return self.threads[tid]

    def init_threads(self):
        """
        初始化不同的变量
        """

        for thread in self.threads:
            thread.sort_event()
            if "MemcpyH2D" in thread.name:
                self.memcpyH2D_thread = thread
            elif "MemcpyD2H" in thread.name:
                self.memcpyD2H_thread = thread
            elif "Steps" in thread.name:
                self.steps_thread = thread
            elif "Scope" in thread.name:
                self.scope_thread = thread
            elif "Ops" in thread.name:
                self.ops_thread = thread
            else:
                logging.INFO(f'unknown thread {thread.name}')

        def MapOps2Scope(self):
            """
             基于时间映射op和scope
            """
            pass

        def MapStep2Scope(self):
            """
            基于时间将每个step(batch)映射到对应的scope
            """
            scope_index = 0
            ops_index = 0

            for step_event in self.scope_thread:
                step_start_time = step_event['ts'], step_end_time = step_event["ts"] + step_event['dur']

        def PassDataFromScope(self):
            pass


class Thread:
    def __init__(self, event):
        # type
        # id
        self.pid = event["args"]
        self.tid = event["tid"]  # thread_id
        self.name = event["args"]["name"]
        self.sorted_index = None
        self.events = []

    def add_event(self, event):
        self.events.append(event)

    def sort_event(self):
        """
        sort based on start time
        """
        self.events.sort(key=lambda x: x["ts"])


def getScores(scope_thread):
    # l = 0
    layer2scopes = {}
    for event in scope_thread.events:
        scope = Scope(event)
        if scope.layer not in layer2scopes.keys():
            layer2scopes[scope.layer] = []
        layer2scopes[scope.layer].append(scope)

    num_layers = len(layer2scopes.keys())
    # sort top scope
    for scopes in layer2scopes.values():
        scopes.sort(key=lambda x: x.start_time)

    for i in range(num_layers-1):
        curLayer = scopes[i], nextLayer = scopes[i+1]
        curIndex = 0, nextIndex = 0sd


class Scope:
    def __init__(self, event):
        self.name = event['name']
        self.start_time = event['ts']
        self.dur = event['dur']
        self.group_id = event['args']['group_id']
        self.layer = event['args']['l']
        self.children = []
