export PYTHONPATH=/workspace/ml_predict/profilers:${PYTHONPATH}
#export PYTHONPATH=/root/guohao/ml_predict/profilers:${PYTHONPATH}

dlprof --mode=pytorch --output_path=../../out/dlprof/conv2d --reports=all --profile_name=conv2d --force=true \
python conv2d.py --batches=1 --image-size=256 --in-channels=3 --out-channels=1 --kernel-size=3 --stride=1 --padding=1 --num-points=1000
