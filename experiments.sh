#!/bin/bash
# data_list=("ispd18_test5_metal5" "ispd18_test8_metal5" "ispd18_test10_metal5" "ispd19_test7_metal5" "ispd19_test8_metal5" "ispd19_test9_metal5")
data_list=("ispd18_test5_metal5")
this_path="/home/weili3/Differentiable-Global-Router" # # DGR path
benchmark_path="/scratch/weili3/cu-gr-2/benchmark"
cugr2="/scratch/weili3/cu-gr-2"
# python data_process_CUGR2.py $cugr2 $benchmark_path # first generate data for DGR
for data in "${data_list[@]}"
do
    # run the whole framework
    echo ""
    echo "Now Run data: $data"
    # echo "Now Run CUGR2 baseline"
    # ./CUGR2.sh $data $benchmark_path $cugr2 $this_path
    cd $this_path
    echo "Now run DGR"
    ./DGR.sh $data $benchmark_path $cugr2 $this_path
done
