"""
An example using pytorch profiler for inference
"""
import torch
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity

if __name__ == "__main__":
    import os

    # os.system("")
    print(os.getenv("PATH"))
    print(os.getenv("LD_LIBRARY_PATH"))

    model = models.resnet18().cuda()
    inputs = torch.randn(5, 3, 224, 224).cuda()

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=True,
                 on_trace_ready=torch.profiler.tensorboard_trace_handler("/root/guohao/ml_predict/tests/log")
                 ) as prof:
        with record_function("model_inference"):
            model(inputs)
    print(str(prof))
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    # prof.export_chrome_trace("/root/guohao/ml_predict/out/traces/inference_trace.json")
