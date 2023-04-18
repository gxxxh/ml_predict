nsys profile -f true --output conv2d -c cudaProfilerApi --capture-range-end=stop --export sqlite --stats=true  --cuda-memory-usage=true python conv2d_test.py
python -m pyprof.parse conv2d.sqlite > conv2d.dict
python -m pyprof.prof conv2d.dict > conv2d.csv