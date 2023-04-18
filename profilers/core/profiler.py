import torch
import logging
from profilers.core.backward import BackwardHelper, backward_available
from profilers.core.autograd import AutogradEngine
import torch.cuda.profiler as profiler
logger = logging.getLogger(__name__)


class OperationProfiler:
    """
    this class is used to profile operation runing
    """

    def __init__(
            self,
            op_name,
            device,
            warm_up=3,
            measure_for=10
    ):
        self.op_name = op_name
        if device!=None:
            self._device = torch.device(device)
        else:
            self._device = torch.device('cpu')
        self._warm_up = warm_up
        self._measure_for = measure_for

    def measure_operation(self, func, args, kwargs):
        func_name = func._get_name()
        for_inplace = _is_potentially_inplace(func_name)

        forward_args, forward_kwargs = self._get_args_for_profiling(
            args, kwargs, for_inplace)

        # We need separate copies of the arguments for the forward and backward
        # measurements because func might be inplace. Running an inplace
        # function repeatedly will affect the autograd graph, which causes
        # problems when we try to measure the backward pass.
        backward_args, backward_kwargs = self._get_args_for_profiling(
            args, kwargs, for_inplace)

        # copy data to device
        func.to(self._device)

        # retval is used for backward
        retval = func(*backward_args, **backward_kwargs)
        if not backward_available(retval):
            logger.info("profile only forward process of {}".format(self.op_name))
            def train_process():
                import torch.cuda.nvtx as nvtx
                nvtx.range_push("layer:Conv")
                # forward
                func(*forward_args, **forward_kwargs)
                nvtx.range_pop()
            self._run_profile(train_process)
        else:
            # logger.info("profile both forward and back process of {}".format(self.op_name))
            print("profile both forward and back process of {}".format(self.op_name))
            def train_process():
                import torch.cuda.nvtx as nvtx
                # forward
                nvtx.range_push("layer:Conv")
                out = func(*forward_args, **forward_kwargs)
                nvtx.range_pop()
                engine = AutogradEngine.new_from(out)
                # backward
                engine.run_backward()
            self._run_profile(train_process)

        return

    def _get_args_for_profiling(self, args, kwargs, for_inplace=False):
        cloned_args = tuple(map(
            lambda arg: self._clone_tensors(arg, for_inplace), args))
        cloned_kwargs = {
            key: self._clone_tensors(value, for_inplace)
            for key, value in kwargs.items()
        }
        return cloned_args, cloned_kwargs

    def _clone_tensors(self, argument, for_inplace):
        if isinstance(argument, torch.Tensor):
            detached = argument.detach()
            detached.requires_grad_(argument.requires_grad)
            # We need to clone the tensor for inplace operations because they
            # cannot be executed on a leaf tensor. This adds some overhead to
            # our backward measurements (an extra CloneBackward function), but
            # it _should_ be negligible. I chose not to exclude CloneBackward
            # from the backward measurements to avoid introducing incorrectness
            # if the user actually uses clone() in their own code.
            return detached.to(self._device) if not for_inplace else detached.clone().to(self._device)

        if isinstance(argument, tuple):
            return tuple(map(
                lambda arg: self._clone_tensors(arg, for_inplace), argument))

        if isinstance(argument, list):
            return list(map(
                lambda arg: self._clone_tensors(arg, for_inplace), argument))

        return argument

    def _run_profile(self, runnable):
        for _ in range(self._warm_up):
            runnable()
        with torch.autograd.profiler.emit_nvtx():
            profiler.start()
            for _ in range(self._measure_for):
                runnable()
            profiler.stop()


# Populated manually from:
# https://pytorch.org/docs/stable/nn.functional.html
POTENTIALLY_INPLACE_FUNCTIONS = {
    'threshold',
    'relu',
    'hardtanh',
    'relu6',
    'elu',
    'selu',
    'celu',
    'leaky_relu',
    'rrelu',
    'dropout',
    'alpha_dropout',
    'dropout2d',
    'dropout3d',

    # In place math operations (+=, *=, -=, /=, //=)
    '__iadd__',
    '__imul__',
    '__isub__',
    '__itruediv__',
    '__ifloordiv__',
}


def _is_potentially_inplace(fn_name):
    return (
        fn_name in POTENTIALLY_INPLACE_FUNCTIONS or
        # In PyTorch, functions with a '_' suffix are in place, by convention
        (len(fn_name) > 1 and fn_name[-1] == '_' and fn_name[-2] != '_')
    )
