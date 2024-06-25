#!/bin/bash
data_list=("ispd18_test8_metal5" "ispd18_test10_metal5" "ispd19_test7_metal5" "ispd19_test8_metal5" "ispd19_test9_metal5")
this_path="/home/weili3/Differentiable-Global-Router"
for data in "${data_list[@]}"
do
    # run the whole framework
    echo ""
    echo "Now Run data: $data"
    echo "Now Run CUGR2 baseline"
    ./CUGR2.sh $data
    cd $this_path
    echo "Now run DGR"
    ./DGR.sh $data
done