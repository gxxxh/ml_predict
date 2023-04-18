"""
 An example for composer to profile training
"""
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from composer import Trainer
from composer.models import mnist_model
from composer.profiler import JSONTraceHandler, cyclic_schedule
from composer.profiler.profiler import Profiler

model = mnist_model()

# Specify Dataset and Instantiate DataLoader
batch_size = 2048
data_directory = '/root/guohao/ml_predict/datasets'

mnist_transforms = transforms.Compose([transforms.ToTensor()])

train_dataset = datasets.MNIST(
    data_directory,
    train=True,
    download=True,
    transform=mnist_transforms)
train_dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=False,
    drop_last=True,
    pin_memory=True,
    persistent_workers=True,
    num_workers=8,
)

# Instantiate the trainer
composer_trace_dir = '/root/guohao/ml_predict/out/composer_profiler'
torch_trace_dir = '/root/guohao/ml_predict/out/torch_profiler'

trainer = Trainer(
    model=model,
    train_dataloader=train_dataloader,
    eval_dataloader=train_dataloader,
    max_duration=2,
    device='gpu' if torch.cuda.is_available() else 'cpu',
    eval_interval=0,
    precision='amp' if torch.cuda.is_available() else 'fp32',
    train_subset_num_batches=16,
    profiler=Profiler(
        trace_handlers=[
            JSONTraceHandler(
                folder=composer_trace_dir,
                overwrite=True)],
        torch_prof_profile_memory=True,
        torch_prof_overwrite=True,
        torch_prof_record_shapes=True,
        torch_prof_with_flops=True,
        torch_prof_with_stack=True,
        schedule=cyclic_schedule(
            wait=0,
            warmup=1,
            active=4,
            repeat=1,
        ),
        torch_prof_folder=torch_trace_dir,
    ))

# https://docs.mosaicml.com/en/latest/trainer/performance_tutorials/analyzing_traces.html
if __name__ == "__main__":
    trainer.fit()
