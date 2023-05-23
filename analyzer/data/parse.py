from tensorflow.python.profiler.internal import _pywrap_profiler
from tensorboard_plugin_profile.convert.raw_to_tool_data import process_raw_trace
import json

from trace_events import *

# Values for type (ph) and s (scope) parameters in catapult trace format.
EVENT_TYPE_METADATA = 'M'
EVENT_TYPE_COMPLETE = 'X'
_TYPE_INSTANT = 'i'
_SCOPE_THREAD = 't'

# thread_name 一个横栏的id
# tid 一个横栏的id

if __name__ == "__main__":
    # trace_path = "/root/guohao/ml_predict/out/logs/conv/plugins/profile/2023_04_27_08_00_50/af11f5fc6e4b.trace.json.gz"
    # raw_data, success = _pywrap_profiler.xspace_to_tools_data([trace_path], "trace_viewer")
    trace_path = "/root/guohao/ml_predict/out/logs/conv/plugins/profile/2023_04_27_08_00_50/af11f5fc6e4b.trace.json"
    data = json.load(open(trace_path))
    print(data.keys())  # displayTimeUnit, metadata, traceEvents
    # init threads
    threads = {}
    for event in data["traceEvents"]:
        if "ph" not in event.keys():
            continue
        if event["ph"] == EVENT_TYPE_METADATA and event["name"] == "thread_name":
            tmp_thread = Thread(event)
            threads[tmp_thread.id] = tmp_thread
    # init steps
    for event in data["traceEvents"]:
        if "ph" not in event.keys():
            continue
        if event["ph"] == EVENT_TYPE_METADATA:
            continue
