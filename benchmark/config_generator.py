import logging
import random

logger = logging.getLogger(__name__)


class OperatorConfigGenerator:
    """
    this class is used to generate class for operator
    """
    def __init__(
            self,
            op_name,
            index_to_config,
            index_filter=None,
    ):
        self._op_name = op_name
        self._index_to_config = index_to_config
        self._index_filter = index_filter
        self._shutdown_early = False

    def add_args(self, parser):
        parser.add_argument('--device', type=str, default='cuda')
        parser.add_argument('--seed', type=int, default=1337)
        parser.add_argument('--num-points', type=int, default=200000)
        # using to  split sample to different worker
        parser.add_argument('--rank', type=int, default=0)
        parser.add_argument('--world-size', type=int, default=1)

    def parse_configurations(self, args, num_configs):
        # Store the arguments for future use
        self._args = args

        if args.rank >= args.world_size:
            raise ValueError('Rank must be less than world size.')
        if args.num_points % args.world_size != 0:
            raise ValueError(
                'Number of points must be divisible by the world size.')
        # Want to ensure we measure the same configurations across each device
        random.seed(args.seed)

        logger.info('Total configurations: %d', num_configs)

        to_record = random.sample(range(num_configs), args.num_points)
        if self._index_filter is not None:
            to_record = list(filter(
                lambda idx: self._index_filter(args, idx),
                to_record,
            ))
            slice_size = len(to_record) // args.world_size
        else:
            slice_size = args.num_points // args.world_size

        logger.info("Total configurations after filtering: %d", len(to_record))
        logger.info("Slice size: %d", slice_size)

        if args.world_size != 1:
            # If we split the sample set across multiple workers, we
            # want to increase the number of overlapping samples between
            # a machine with just one worker if this recording script is
            # stopped early. This is because the workers process the
            # configurations sequentially.
            random.shuffle(to_record)
            offset = slice_size * args.rank
            to_record = to_record[offset:offset + slice_size]
        configs = []
        for i in range(0, len(to_record)):
            config = self._index_to_config(args, to_record[i])
            configs.append(config)
        return configs
