import argparse
import logging
import math
from benchmark.config_generator import OperatorConfigGenerator

logger = logging.getLogger(__name__)

MIN_IN_CHANNELS = 3
MIN_OUT_CHANNELS = 16


def index_to_config(args, index):
    bias = False if index % 2 == 0 else True
    index //= 2

    batch = (index % args.batches) + 1
    index //= args.batches

    image_size = (index % args.image_size) + 1
    index //= args.image_size

    in_channels = (index % args.in_channels) + 1
    index //= args.in_channels

    out_channels = (index % args.out_channels) + 1
    index //= args.out_channels

    kernel_size = (index % args.kernel_size) + 1
    index //= args.kernel_size

    stride = (index % args.stride) + 1
    index //= args.stride

    # Padding is 0-based
    padding = index

    return (
        bias,
        batch,
        image_size,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
    )


def index_filter(args, index):
    config = index_to_config(args, index)
    # NOTE: We multiply because the dimensions have different ranges; we want
    #       them to each "contribute equally". We weigh the image size more to
    #       select smaller image sizes.
    # image_size (1-dim) * in_channels * out_channels * kernel_size
    conv_size = math.pow(config[2], 1.15) * config[3] * config[4] * config[5]

    # padded_input_size should larger than kernel size
    padded_input_size = config[2] + 2 * config[7]
    kernel_size = config[5]
    # NOTE: This value was chosen arbitrarily: we don't want the in/out
    #       channels and image size to all be too large. This way, large values
    #       for the in/out channels would lead to a smaller image size (and
    #       vice versa).
    return conv_size <= 35000000 and config[3] >= MIN_IN_CHANNELS and config[4] >= MIN_OUT_CHANNELS and padded_input_size >= kernel_size


def main():
    configGenerator = OperatorConfigGenerator(
        op_name='conv2d',
        index_to_config=index_to_config,
        index_filter=index_filter,
    )

    parser = argparse.ArgumentParser()
    configGenerator.add_args(parser)
    # todo batches should between [1,2,4,8,16,32,64,128,256,512,1024]
    parser.add_argument('--batches', type=int, default=64)
    parser.add_argument('--image-size', type=int, default=256)
    parser.add_argument('--in-channels', type=int, default=2048)
    parser.add_argument('--out-channels', type=int, default=2048)
    parser.add_argument('--kernel-size', type=int, default=11)
    parser.add_argument('--stride', type=int, default=4)
    # Padding is 0-based, so this means we consider 0 to 3 inclusive
    parser.add_argument('--padding', type=int, default=4)
    # config save path
    parser.add_argument('--save', type=str, default="conv2d_configs.txt")
    args = parser.parse_args()

    num_configs = (
        2 *  # Whether or not there is a bias
        args.batches *
        args.image_size *
        args.in_channels *
        args.out_channels *
        args.kernel_size *
        args.stride *
        args.padding
    )

    # Conv2d has filtering, so we won't have exactly 200000 points (the
    # default). So here we increase the number of starting points.
    args.num_points *= 6
    configs = configGenerator.parse_configurations(args, num_configs)
    logger.info("Generated %d configurations", len(configs))
    with open(args.save, "w") as f:
        for config in configs:
            f.write(config+ '\n')
        f.close()


if __name__ == '__main__':
    kwargs = {
        "format": "%(asctime)s %(levelname)-8s %(message)s",
        "datefmt": "%Y-%m-%d %H:%M",
        "level": logging.INFO,
    }
    logging.basicConfig(**kwargs)
    main()
