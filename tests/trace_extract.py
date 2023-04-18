import json

import gzip
import json
import os
import unittest

from torch_tb_profiler.profiler.data import (DistributedRunProfileData,
                                             RunProfileData)
from torch_tb_profiler.profiler.loader import RunLoader
from torch_tb_profiler.profiler.overall_parser import ProfileRole
from torch_tb_profiler.profiler.gpu_metrics_parser import GPUMetricsParser
from torch_tb_profiler.run import RunProfile
def parse_json_trace(file_name):
    with open(file_name, "r") as f:
        json_content = f.read()
    trace_json = json.loads(json_content)
    return RunProfileData.from_json("worker0", 0, trace_json)

if __name__=="__main__":
    from torch_tb_profiler.profiler import trace
    from torch_tb_profiler.profiler.module_op import (
        _build_module_hierarchy, aggegate_module_view)
    data = parse_json_trace("/root/guohao/ml_predict/tests/log/dell05_6691.1679993279479.pt.trace.json")
    stats = aggegate_module_view(data.tid2tree, data.events)
    data.process()
