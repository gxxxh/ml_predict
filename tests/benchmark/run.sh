nsys profile -f true --output conv2d -c cudaProfilerApi --capture-range-end repeat:5 --export sqlite --cuda-memory-usage=true python conv2d_test.py
python -m pyprof.parse --memory true --file conv2d.sqlite > conv2d.dict
python -m pyprof.prof conv2d.dict > conv2d.csv
python -m pyprof.prof conv2d.dict --output conv2d.csv