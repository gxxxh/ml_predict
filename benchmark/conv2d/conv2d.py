import argparse
import logging
import torch
from profilers.core.profiler import OperationProfiler
import pyprof

logger = logging.getLogger(__name__)

MIN_IN_CHANNELS = 3
MIN_OUT_CHANNELS = 16

torch.backends.cudnn.benchmark = True

def config_to_profiler_args(bias,
                            batch,
                            image_size,
                            in_channels,
                            out_channels,
                            kernel_size,
                            stride,
                            padding):

    device = torch.device('cuda')
    conv2d = torch.nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=bias,
    )
    inp = torch.randn((
        batch,
        in_channels,
        image_size,
        image_size,
    ))
    # NOTE: This is important: for most convolutions, we will also need the
    #       gradient with respect to the input to be able to backpropagate to
    #       earlier operations in the network.
    inp = inp.requires_grad_()

    return {
        'func': conv2d,
        'args': (inp,),
        'kwargs': {},
    }


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--bias', type=bool)
    parser.add_argument('--batch', type=int)
    parser.add_argument('--image_size', type=int)
    parser.add_argument('--in_channels', type=int)
    parser.add_argument('--out_channels', type=int)
    parser.add_argument('--kernel_size', type=int)
    parser.add_argument('--stride', type=int)
    parser.add_argument('--padding', type=int)
    args = parser.parse_args()
    kwargs = config_to_profiler_args(**vars(args))
    if kwargs is None:
        logger.info('no proper config')
        return
    profiler = OperationProfiler(device='cuda', op_name="conv2d")
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
    main()
