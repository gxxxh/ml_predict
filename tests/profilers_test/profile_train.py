"""
An example for profile training using pytorch profiler
"""
import pathlib
import torch

# RESNET50_BATCHES = [16, 32, 64]
RESNET50_BATCHES = [16]

trace_path = str(pathlib.Path(__file__).parent.parent / "out" / "traces")


def trace_handler(prof, ):
    print(trace_path)
    prof.export_chrome_trace(
        trace_path + "/test_res_" + str(prof.step_num) + ".json")


def run_resnet50():
    import models.resnet.entry_point as rep
    model = rep.skyline_model_provider()
    iteration = rep.skyline_iteration_provider(model)

    for batch_size in RESNET50_BATCHES:
        inputs = rep.skyline_input_provider(batch_size)

        def runnable():
            iteration(*inputs)

        N = 10
        with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA],
                record_shapes=True,
                profile_memory=True,
                with_stack=True,

                # In this example with wait=1, warmup=1, active=2,
                # profiler will skip the first step/iteration,
                # start warming up on the second, record
                # the third and the forth iterations,
                # after which the trace will become available
                # and on_trace_ready (when set) is called;
                # the cycle repeats starting with the next step

                schedule=torch.profiler.schedule(
                    wait=2,
                    warmup=3,
                    active=N - 5),
                # on_trace_ready=trace_handler
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    "/root/guohao/ml_predict/tests/log")
                # used when outputting for tensorboard
        ) as p:
            for iter in range(N):
                runnable()
                # send a signal to the profiler that the next iteration has
                # started
                p.step()
            p.events()
        # trace_handler(p)


if __name__ == "__main__":
    run_resnet50()
