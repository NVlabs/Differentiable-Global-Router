#!/bin/bash
# data_list=("ispd18_test1" "ispd18_test5_metal5" "ispd18_test8" "ispd18_test8_metal5")
data_list=("ispd18_test1" "ispd18_test5" "ispd18_test5_metal5" "ispd18_test8")
# data_list=("ispd18_test8" "ispd18_test8_metal5" "ispd19_test2" "ispd19_test7" "ispd19_test7_metal5")
# data_list=("ispd19_test2" "ispd19_test7" "ispd19_test7_metal5" "ispd19_test8_metal5" "ispd19_test9_metal5")
this_path="/home/wli1/differentiable-global-routing"
benchmark_path="/home/scratch.rliang_hardware/wli1/cu-gr/benchmark"
cugr2="/home/scratch.rliang_hardware/wli1/cu-gr-2"
this_path="/home/wli1/differentiable-global-routing"
for data in "${data_list[@]}"
do
    # run the whole framework
    # echo new line
    echo ""
    echo "DRCU: Run data: $data"
    cd $cugr2
    # output2 is CUGR2
    ./drcu -lef $benchmark_path/$data/$data.input.lef -def $benchmark_path/$data/$data.input.def -thread 8 -guide $benchmark_path/$data/$data.output2 --output $cugr2/DR_result/CUGR2_$data.txt --tat 2000000000 > $cugr2/DR_log/$data\_CUGR2.log
    # output3 is Ours
    # ./drcu -lef $benchmark_path/$data/$data.input.lef -def $benchmark_path/$data/$data.input.def -thread 8 -guide $benchmark_path/$data/$data.output3 --output $cugr2/DR_result/Ours_$data.txt --tat 2000000000 > $cugr2/DR_log/$data\_Ours.log
    # output5 is Ours_w.CZ
    ./drcu -lef $benchmark_path/$data/$data.input.lef -def $benchmark_path/$data/$data.input.def -thread 8 -guide $benchmark_path/$data/$data.output5 --output $cugr2/DR_result/Ours_$data.txt --tat 2000000000 > $cugr2/DR_log/$data\_Ours_CZ.log
    # ./New_Tree.sh $data
    # cd $this_path
done