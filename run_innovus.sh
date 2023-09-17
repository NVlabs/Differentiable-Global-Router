#!/bin/bash

# After obtaining DR result, we need to evaluate using innovus

data="$1"
method="$2"
benchmark_path="/scratch/weili3/cu-gr-2/benchmark"
cugr2="/scratch/weili3/cu-gr-2"
this_path="/home/weili3/Differentiable-Global-Router"
eval_path="/scratch/weili3"

export LD_LIBRARY_PATH=""
if [[ "$data" == *"ispd18"* ]]; then
    benchmark="ispd18"
else
    benchmark="ispd19"
fi

cd "$eval_path/$benchmark""eval/"

if [ "$benchmark" == "ispd18" ]; then
    # cd "$eval_path/$benchmark""eval/

    bash "$eval_path/$benchmark""eval/$benchmark""eval.sh" -lef "$benchmark_path/$data/$data.input.lef" -def "$cugr2/DR_result/${method}_$data.txt" -guide "$benchmark_path/$data/$data.${method}.guide" -thread 16 |& tee "$benchmark_path/$data/${method}_$data.final.log"
else
    bash "$eval_path/$benchmark""eval/$benchmark""eval.sh" -lef "$benchmark_path/$data/$data.input.lef" -idef "$benchmark_path/$data/$data.input.def" -odef "$cugr2/DR_result/${method}_$data.txt" -guide "$benchmark_path/$data/$data.output3" |& tee "$benchmark_path/$data/${method}_$data.final.log"
fi
