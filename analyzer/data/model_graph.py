import keras.engine.input_layer

from analyzer.data.util import TimeItem


class Node:
    def __init__(self):
        self.forward_start = 10e9
        self.forward_duration = 0
        self.backward_start = 10e9
        self.backward_duration = 0
        self.name = ""

    def add_forward(self, t):
        self.forward_duration += t.duration
        self.forward_start = min(self.forward_start, t.get_start_time())

    def add_backward(self, t):
        self.backward_duration += t.duration
        self.backward_start = min(self.backward_start, t.get_start_time())

    def get_name(self):
        return self.name


class LayerNode(Node):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        self.name = self.layer.name

    def get_input(self):
        self.layer.input.shape

    def get_output(self):
        self.layer.output.shape


class OptimizerNode(Node):
    def __init__(self, optimizer):
        super().__init__()
        self.optimizer = optimizer
        if self.optimizer != None:
            self.name = self.optimizer.__class__.__name__
        self.optimizer_init_duration = 0  # init time, before forward
        self.optimizer_duration = 0  # run time, during backward

    def add_init_time(self, t):
        self.optimizer_init_duration += t.duration

    def add_run_time(self, t):
        self.optimizer_duration += t.duration


class LossNode(Node):
    def __init__(self, loss_name):
        super().__init__()
        self.name = loss_name


class ModelGraph:
    def __init__(self):
        self.layers = []  # type 为node
        self.inputLayer = None
        self.loss = None
        self.optimizer = None

    def parseModel(self, model):
        """
        转化为Layer
        """
        self.loss = LossNode(model.loss)
        self.optimizer = OptimizerNode(model.optimizer)
        self.parseLayers(model)

    def parseLayers(self, model):
        for layer in model.layers:
            if hasattr(layer, 'layers'):
                self.parseLayers(layer)
            elif isinstance(layer, keras.engine.input_layer.InputLayer):
                self.inputLayer = layer
            else:
                self.layers.append(LayerNode(layer))

    def get_layers(self):
        return self.layers

    def get_nodes(self):
        name2node = {}
        for layer in self.layers:
            name2node[layer.name] = layer
        return name2node

    def print_layers(self):
        for layer in self.layers:
            print(layer.get_name())


def get_model_graph(model):
    model_graph = ModelGraph()
    model_graph.parseModel(model)
    return model_graph
