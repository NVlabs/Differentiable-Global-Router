#!/bin/bash
# data_list=("ispd18_test5" "ispd18_test5_metal5" "ispd18_test8" "ispd18_test8_metal5" "ispd18_test10" "ispd18_test10_metal5" "ispd19_test7" "ispd19_test8" "ispd19_test9" "ispd19_test7_metal5" "ispd19_test8_metal5" "ispd19_test9_metal5")
# data_list=("ispd18_test5" "ispd18_test5_metal5" "ispd18_test10_metal5" "ispd19_test7_metal5" )
data_list=("ispd18_test5_metal5")
# data_list=("ispd18_test8" "ispd18_test8_metal5" "ispd19_test2" "ispd19_test7" "ispd19_test7_metal5")
# data_list=("ispd19_test7_metal5" "ispd19_test8_metal5" "ispd19_test9_metal5")
this_path="/home/weili3/Differentiable-Global-Router"
for data in "${data_list[@]}"
do
    # run the whole framework
    # echo new line
    cd $this_path
    ./ours_CZ.sh $data
done