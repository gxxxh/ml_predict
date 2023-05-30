import json
import logging

from analyzer.data.trace_events import ScopeItem, TrainStep, get_process
from analyzer.data.util import time_in, time_before, TimeItem


# todo flatten no time
def parse_trace(file_name):
    trace_data = None
    with open(file_name, 'r') as f:
        trace_data = json.load(f)
    process = get_process(trace_data)
    train_steps = MapSteps2Scope(process.steps_thread, process.scope_thread)
    return train_steps


def MapSteps2Scope(steps_thread, scope_thread):
    """
    基于时间将每个step(batch)映射到对应的scope
    """

    train_steps = [TrainStep(event) for event in steps_thread.events]
    for event in scope_thread.events:
        for i in range(len(train_steps)):
            scope = ScopeItem(event)
            if time_in(scope.time, train_steps[i].time):
                train_steps[i].add_scope(scope)
    for train_step in train_steps:
        train_step.parseEvents()
    # sort based on start time
    train_steps.sort(key=lambda x: x.get_start_time())
    # 删除第一个和最后一个
    return train_steps[1:-1]


class Analyzer:
    def __init__(self, model_graph, train_step):
        self.model_graph = model_graph
        self.train_step = train_step
        self.name = "analyzer"
        self.forward_start = -1  # idx in scopes for the first layer's forward
        self.loss_start = -1
        self.backward_start = -1
        self.name2nodes = None
        self.forward_times = []
        self.backward_times = []
        self.loss_times = []
        self.optimizer_init = []
        self.optimizer_backward = []

    def do_analyze(self):
        self.name2nodes = self.model_graph.get_nodes()
        self.init_idx()
        self._analyze_forward()
        self._anaylze_backward()
        # loss_start为-1
        self._analyze_loss()
        self._analyze_optimizer()

    def map_node_time(self, run_times, type):
        """
        配合_parse_layer_info
        不再使用
        """
        for time_item in run_times:
            if time_item[0] not in self.name2node.keys():
                logging.warning(f"no layer for name {time_item[0]}")
                continue
            else:
                if type == 'forward':
                    self.name2node[time_item[0]].add_forward(time_item[1])
                else:
                    self.name2node[time_item[0]].add_backward(time_item[1])

    def _parse_layer_info(self, scope):
        # 广搜获取某一层的资源
        # 前向传播获取最低层，反向传播获取倒数第一层
        # not working
        depth = scope.depth
        if scope.is_backward():
            depth -= 1
        curDepth = 1
        q = [scope]
        while curDepth < depth:
            curDepth += 1
            sz = len(q)
            for i in range(sz):
                curScope = q[0]
                q.pop(0)
                for child in curScope.children:
                    q.append(child)
        return [[scope.name, scope.time] for scope in q]

    def _bfs(self, scope, type):
        # 通过bfs获取所需的scope对象，根据名称匹配
        q = [scope]
        while len(q) > 0:
            curScope = q[0]
            q.pop(0)
            if curScope.name in self.name2nodes.keys():
                if type == 'backward':
                    self.name2nodes[curScope.name].add_backward(curScope.time)
                else:
                    self.name2nodes[curScope.name].add_forward(curScope.time)
            for child in curScope.children:
                q.append(child)

    def _merge_layer_times(selfs, times):
        res = []
        i = 0
        while i < len(times):
            k = 1
            cur = times[i]
            while i + k < len(times):
                if cur[0] == times[i + k][0]:
                    cur[1].merge(times[i + k][1])
                    k += 1
                else:
                    break
            res.append(cur)
            i = i + k
        return res

    def init_idx(self):
        """
        找到流程的边界
        """
        layers = self.model_graph.get_layers()
        scopes = self.train_step.get_scope_trees()
        # init_forward
        i = 0
        while i < len(scopes):
            layer_time = scopes[i].find(layers[0].name)
            if layer_time != None and scopes[i].is_backward() == False:
                self.forward_start = i
                break
            else:
                i += 1
        if self.forward_start == -1:
            logging.warning(f"no start for trace, start layer name is {layers[0].name}")

        while i < len(scopes):
            t = scopes[i].find(self.model_graph.loss.get_name())
            if scopes[i].is_backward() == False and t != None:
                self.loss_start = i
                break
            i += 1
        if self.loss_start == -1:
            logging.warning(f"no loss for trace, loss  name is {self.model_graph.loss}")

        i = self.forward_start
        while i < len(scopes):
            if scopes[i].is_backward():
                # if scopes[i].find(self.model_graph.loss) or scopes[i].find(layers[-1].name):
                if scopes[i].find(layers[-1].get_name()):
                    self.backward_start = i
                    break
            i += 1
        if self.backward_start == -1:
            logging.warning(f"no backward for trace, backward  name is {layers[-1].name}")

    def _analyze_forward(self):
        scopes = self.train_step.get_scope_trees()
        # map forward
        i = self.forward_start
        while i < self.loss_start:
            if scopes[i].is_backward() == False:
                self._bfs(scopes[i], 'forward')
            i += 1

    def _anaylze_backward(self):
        scopes = self.train_step.get_scope_trees()
        i = self.backward_start
        while i < len(scopes):
            if scopes[i].is_backward():
                self._bfs(scopes[i], 'backward')
            i += 1

    def _analyze_loss(self, ):
        scopes = self.train_step.get_scope_trees()
        # map loss
        i = self.forward_start
        while i < self.backward_start:
            t = scopes[i].find(self.model_graph.loss.get_name())
            if t != None:
                if scopes[i].is_backward():
                    self.model_graph.loss.add_backward(t)
                else:
                    self.model_graph.loss.add_forward(t)
            i += 1

    def _analyze_optimizer(self):
        scopes = self.train_step.get_scope_trees()
        i = 0
        while i < self.forward_start:
            t = scopes[i].find(self.model_graph.optimizer.get_name())
            if t != None:
                self.model_graph.optimizer.add_init_time(t)
            i += 1
        i = self.backward_start
        while i < len(scopes):
            t = scopes[i].find(self.model_graph.optimizer.get_name())
            if t != None:
                self.model_graph.optimizer.add_run_time(t)
            i += 1

# def aggregate(dir):
#     # Aggregate all the csv files in the directory
#     # and return a dataframe
#     for root, _, files in os.walk(dir):
#         i = 0
#         for file in files:
#             if file.endswith(".csv"):
#                 file_path = os.path.join(root, file)
#                 df = pd.read_csv(file_path)
#                 df["file"] = file_path
#                 if 'df_all' not in locals():
#                     df_all = df
#                 else:
#                     df_all = pd.concat([df_all, df], ignore_index=True)
#             i += 1
#             if i % 50 == 0:
#                 print("forward kernel " + str(len(df_all[df_all["dir"] == "fprop"]["kernel"].unique())))
#                 print("backward kernel " + str(len(df_all[df_all["dir"] == "bprop"]["kernel"].unique())))
#     return df_all
