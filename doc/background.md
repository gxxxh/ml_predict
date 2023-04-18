## profiler

| name             | layer info | operator info | kernel info | layer-kernel map | layer operator map | operator input | kernel input | url                                                                                                                                                                             |
|------------------|------------|---------------|-------------|------------------|--------------------|----------------|--------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| pytorch profiler | √          | √             | √           | ×                | ×                  | ×              | ×            |
| composer         | √          | √             | √           | ×                | ×                  | ×              | ×            | [doc](https://github.com/mosaicml/composer)                                                                                                                                     |
| nvidia-dlprof    | √          | √             | √           | √                | ×                  | ×              | ×            | [tutorial](https://tigress-web.princeton.edu/~jdh4/how_to_profile_with_dlprof_may_2021.pdf),[doc](https://docs.nvidia.com/deeplearning/frameworks/dlprof-user-guide/index.html) |
| pyperf           | √          | √             |             |                  |                    |                |              | https://github.com/NVIDIA/PyProf)                                                                                                                                               |
| pyperf2          | √          | √             |             |                  |                    |                |              | [doc](https://github.com/adityaiitb/pyprof2)                                                                                                                                    |

## nvidia docker

### pytorch

```bash
docker run --rm --gpus=1 --shm-size=1g --ulimit memlock=-1 \
--ulimit stack=67108864 -it -p 8000:8000 -v /root/guohao/ml_predict:/work_space/ml_predict \
nvcr.io/nvidia/pytorch:20.07-py3
```

## benchmark
