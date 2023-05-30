# this file define class in profile json

import logging

from analyzer.data.util import TimeItem, time_in, time_before

EVENT_TYPE_METADATA = 'M'
EVENT_TYPE_COMPLETE = 'X'

THREAD_TYPE_STEPS = "Steps"
THREAD_TYPE_SCOPE = "TensorFlow Name Scoep"
THREAD_TYPE_OPS = "TensorFolw Ops"
SCOPE_TYPE_FORWARD = "sequencial"
SCOPE_TYPE_LOSS = "loss"
SCOPE_TYPE_BACKWARD = "gradient_tape"


def get_process(trace_data):
    events = trace_data['traceEvents']
    gpu_process = Process()
    # init gpu process
    for event in events:
        if len(event.keys()) == 0:
            continue
        if event['ph'] == EVENT_TYPE_METADATA and event['name'] == 'process_name':
            if 'device' in event['args']['name']:
                gpu_process.parse_process_event(event)
                break
    # init thread
    for event in events:
        if len(event.keys()) == 0:
            continue
        if event['ph'] == EVENT_TYPE_METADATA and event['name'] == 'thread_name':
            gpu_process.add_meta_event(event)

    for event in events:
        if len(event.keys()) == 0:
            continue
        if event['ph'] == EVENT_TYPE_COMPLETE:
            gpu_process.add_type_event(event)
    gpu_process.init_threads()
    return gpu_process


class Process:
    """
    A Process for CPU of GPU
    """

    def __init__(self):
        """
        device:GPU or host CPU
        """
        self.pid = None
        self.name = None
        self.threads = {}
        self.memcpyH2D_thread = None
        self.memcpyD2H_thread = None
        self.steps_thread = None
        self.scope_thread = None
        self.ops_thread = None

    def parse_process_event(self, event):
        self.pid = event["pid"]
        self.name = event["args"]["name"]

    def valid_event(self, event):
        """
        check if the event belong to this process
        """
        if (len(event.keys())) == 0:
            return False
        pid = event['pid']
        if pid != self.pid:
            return False
        return True

    def add_meta_event(self, event):
        """
        init all threads
        """
        if self.valid_event(event) == False:
            return
        threadid = event['tid']
        if threadid not in self.threads.keys():
            thread = Thread(event)
            self.threads[thread.tid] = thread

    def add_type_event(self, event):
        """
        add event to thread
        """
        if self.valid_event(event) == False:
            return
        threadid = event['tid']
        if threadid not in self.threads.keys():
            logging.warning(f'no thread for event: {event}')
        self.threads[threadid].add_event(event)

    def get_thread_by_id(self, tid):
        if tid not in self.threads.keys():
            return None
        return self.threads[tid]

    def init_threads(self):
        """
        init thread we need
        """

        for thread in self.threads.values():
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
                logging.info(f'unknown thread {thread.name}')


class Thread:
    """
    Thread contains events sorted by time without conflict
    """

    def __init__(self, event):
        self.pid = event["args"]
        self.tid = event["tid"]  # thread_id
        self.name = event["args"]["name"]
        self.sorted_index = None
        self.events = []
        # self.time = TimeItem().init_by_event(event)

    def add_event(self, event):
        self.events.append(event)

    def sort_event(self):
        """
        sort based on start time
        """
        self.events.sort(key=lambda x: x["ts"])

    def get_event(self, i):
        return self.events[i]

    def events_num(self):
        return len(self.events)


class ScopeItem:
    """
    define a scope in the scope thread
    socpe was formatted in tree struct
    """

    def __init__(self, event):
        self.name = event['name']
        self.time = TimeItem().init_by_event(event)
        self.group_id = event['args']['group_id']
        self.layer = event['args']['l']
        self.children = []
        self.depth = 1

    def add_child(self, scopeItem):
        self.children.append(scopeItem)
        self.depth = max(self.depth, scopeItem.depth + 1)
        self.children.sort(key=lambda x: x.get_start_time())

    def get_start_time(self):
        return self.time.start_time

    def find(self, layer_name):
        if (self.name == layer_name):
            return self.time
        for child in self.children:
            if (child.find(layer_name) != None):
                return child.find(layer_name)
        return None

    def is_backward(self):
        return self.find(SCOPE_TYPE_BACKWARD) != None


class TrainStep:
    """
    define a train step,
    """

    def __init__(self, event):
        self.event = event
        self.time = TimeItem().init_by_event(event)
        self.scopes = []
        self.trees = []  # format scopes into trees, whose's root are socpe in level 0
        self.MemcpyH2D = None
        self.MemcpyD2H = None

    def add_scope(self, scope):
        self.scopes.append(scope)

    def get_start_time(self):
        return self.time.get_start_time()

    def parseEvents(self):
        """
        将event转化为树结构
        """
        scopesDict = {}
        # 基于层对时间分类
        for scope in self.scopes:
            if scope.layer not in scopesDict:
                scopesDict[scope.layer] = []
            scopesDict[scope.layer].append(scope)
        for scopes in scopesDict.values():
            scopes.sort(key=lambda x: x.get_start_time())

        layers = scopesDict.keys()
        layers = sorted(layers, reverse=True)
        # 子层添加到父层中
        for k in range(len(layers) - 1):
            curLayerEvents = scopesDict[layers[k]]
            parentLayerEvents = scopesDict[layers[k + 1]]
            i = j = 0
            while i < len(curLayerEvents) and j < len(parentLayerEvents):
                curEvent = curLayerEvents[i]
                parentEvent = parentLayerEvents[j]
                if time_in(curEvent.time, parentEvent.time):
                    parentLayerEvents[j].add_child(curLayerEvents[i])
                    i+=1
                elif time_before(parentEvent.time, curEvent.time):
                    j += 1
                else :
                    logging.warning(f"no parent for event {curEvent.time}, parent {parentEvent.time}")
                    i+=1
        # 返回layer[0]
        self.trees = scopesDict[min(layers)]

    def get_scope_trees(self):
        return self.trees
