#!/bin/bash
# data_list=("ispd19_test8_metal5" "ispd19_test9_metal5" "ispd19_test7" "ispd19_test8" "ispd19_test9")
conda activate 118
# data_list=("ispd18_test5" "ispd18_test5_metal5" "ispd18_test8_metal5" "ispd18_test10_metal5" "ispd19_test7_metal5" "ispd19_test8_metal5" "ispd19_test9_metal5")
# data_list=("ispd18_test5_metal5")
data_list=("ispd18_test5")
# data_list=("ispd18_test5" "ispd18_test8" "ispd18_test10" "ispd19_test7" "ispd19_test8" "ispd19_test9")
# data_list=("ispd18_test5" "ispd18_test8" "ispd18_test9" "ispd18_test10" "ispd18_test5_metal5" "ispd18_test8_metal5" "ispd18_test10_metal5")
this_path="/home/weili3/Differentiable-Global-Router" # # DGR path
benchmark_path="/scratch/weili3/cu-gr-2/benchmark"
cugr2="/scratch/weili3/cu-gr-2"
# python data_process_CUGR2.py $cugr2 $benchmark_path # first generate data for DGR
for data in "${data_list[@]}"
do
    # run the whole framework
    # echo new line
    echo ""
    echo "Now Run data: $data"
    # ./ours_NoVia.sh $data $benchmark_path $cugr2 $this_path
    # echo "No via result run, now run w. via (original one) but the bug is fixed now..."
    # cd $this_path
    ./CUGR2.sh $data $benchmark_path $cugr2 $this_path
    # ./ours.sh $data $benchmark_path $cugr2 $this_path
    # echo "Our results obtained, now run ours w. multiple tree..."
    # cd $this_path
    # ./New_Tree.sh $data $benchmark_path $cugr2 $this_path
    # echo "Multiple tree obtained, now run ours w. CZ ..."
    # cd $this_path
    # ./ours_CZ.sh $data $benchmark_path $cugr2 $this_path
done
# print final innovus result
python read_stat.py $benchmark_path