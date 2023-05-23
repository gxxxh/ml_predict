"""
An exampel to test torch autograd profiler
"""
import pathlib
import torch

# RESNET50_BATCHES = [16, 32, 64]
RESNET50_BATCHES = [16]

trace_path = str(pathlib.Path(__file__).parent.parent / "out" / "traces")


def trace_handler(prof, ):
    print(trace_path)
    prof.export_chrome_trace(
        trace_path + "/test_autograd_" + str(prof.step_num) + ".json")


def run_resnet50():
    import models.resnet.entry_point as rep
    model = rep.skyline_model_provider()
    iteration = rep.skyline_iteration_provider(model)

    for batch_size in RESNET50_BATCHES:
        inputs = rep.skyline_input_provider(batch_size)

        def runnable():
            iteration(*inputs)

        N = 5
        with torch.cuda.profiler.profile():
            for i in range(5):
                runnable()
            with torch.autograd.profiler.emit_nvtx():
                for _ in range(2):
                    runnable()


if __name__ == "__main__":
    run_resnet50()
