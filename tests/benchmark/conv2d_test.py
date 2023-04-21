import logging
import pyprof
from benchmark.conv2d.conv2d import config_to_profiler_args
from profilers.core.profiler import OperationProfiler
logger = logging.getLogger(__name__)
def test_benchmark(i):

    kwargs = config_to_profiler_args(
        bias=True,
        batch=16,
        image_size=54,
        in_channels=956,
        out_channels=17,
        kernel_size=2,
        stride=3,
        padding=1)
    profiler = OperationProfiler(op_name=f'Conv2d{i}', device='cuda')
    profiler.measure_operation(**kwargs)


if __name__ == '__main__':
    kwargs = {
        "format": "%(asctime)s %(levelname)-8s %(message)s",
        "datefmt": "%Y-%m-%d %H:%M",
        "level": logging.INFO,
    }
    logging.basicConfig(**kwargs)
    logger.info("Initializing PyProf...")
    pyprof.init()
    # test_benchmark(0)
    for i in range(5):
        logger.info(f"Start Profile {i}")
        test_benchmark(i)
        import torch
        import gc
        gc.collect()
        torch.cuda.empty_cache()

