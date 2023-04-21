# python environment
export PYTHONPATH=/root/guohao/ml_predict:${PYTHONPATH}
# variables
configFile=conv2d_configs.txt
BenchmarkPythonScript=conv2d.py
ResultSavePath=/root/guohao/ml_predict/out/benchmark/conv2d
i=0
cat ${configFile} | while read line; do
  resultName="conv2d${i}"
  echo "Running config${i}: ${line}"
  pythonVars=($(echo $line | tr ',' ' '))
  # run the benchmark
  nsys profile \
    -f true --output $resultName -c cudaProfilerApi --capture-range-end stop --export sqlite --cuda-memory-usage true \
    python ${BenchmarkPythonScript} \
    --bias ${pythonVars[0]} --batch ${pythonVars[1]} --image_size ${pythonVars[2]} --in_channels ${pythonVars[3]} \
    --out_channels ${pythonVars[4]} --kernel_size ${pythonVars[5]} --stride ${pythonVars[6]} --padding ${pythonVars[7]}

  # move nsys result to output directory
  echo "moving nsys result file to ${ResultSavePath}"
  mv ${resultName}.sqlite ${ResultSavePath}
  mv ${resultName}.nsys-rep ${ResultSavePath}

  # parse data
  python -m pyprof.parse --memory true --file ${ResultSavePath}/${resultName}.sqlite >$ResultSavePath/${resultName}.dict
  python -m pyprof.prof ${ResultSavePath}/${resultName}.dict --output ${ResultSavePath}/${resultName}.csv
#  if ((i==10));then
#    break
#  fi
  ((i++))
done
