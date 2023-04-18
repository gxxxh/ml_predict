import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.profiler as profiler
import torch.optim as optim

import pyprof
from benchmark.conv2d.conv2d import config_to_profiler_args
from profilers.core.profiler import OperationProfiler

def main():

    kwargs = config_to_profiler_args(
        bias=True,
        batch=16,
        image_size=54,
        in_channels=956,
        out_channels=17,
        kernel_size=2,
        stride=3,
        padding=1)
    profiler = OperationProfiler(op_name='conv2d', device='cuda')
    profiler.measure_operation(**kwargs)


if __name__ == '__main__':
    kwargs = {
        "format": "%(asctime)s %(levelname)-8s %(message)s",
        "datefmt": "%Y-%m-%d %H:%M",
        "level": logging.INFO,
    }
    logging.basicConfig(**kwargs)
    main()
