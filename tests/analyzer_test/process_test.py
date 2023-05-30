from analyzer.data.model_graph import ModelGraph, get_model_graph
from analyzer.data.process import parse_trace, Analyzer
from tf_models.models.model_factory import get_model
from tensorflow.keras.optimizers import Adam

if __name__ == '__main__':
    model = get_model('resnet50')
    model.compile(
        optimizer=Adam(
        learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    model_graph = get_model_graph(model)
    # model_graph.print_layers()

    trace_json = '/root/guohao/ml_predict/out/logs/resnet/plugins/profile/2023_05_24_07_55_43/cbf414b4c396.trace.json'
    train_steps = parse_trace(trace_json)
    for train_step in train_steps:
        analyzer = Analyzer(model_graph, train_step)
        analyzer.do_analyze()
