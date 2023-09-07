# run different parameters for result analysis
data="${1:-ispd18_test5_metal5}"
benchmark_path="/scratch/weili3/cu-gr-2/benchmark"
cugr2="/scratch/weili3/cu-gr-2"
this_path="/home/weili3/Differentiable-Global-Router"
echo "Processing Ours w. CZ Round... data: $data"
python3 main.py --data_path $cugr2/run/$data.pt --add_CZ True
cd $cugr2/run/
./route -lef $benchmark_path/$data/$data.input.lef -def $benchmark_path/$data/$data.input.def -output $benchmark_path/$data/$data.output5 -dgr $this_path/CUGR2_guide/CUgr_$data\_0\_1.txt > $cugr2/GR_log/$data\_Ours_CZ.log
# cd $cugr2
# ./drcu -lef $benchmark_path/$data/$data.input.lef -def $benchmark_path/$data/$data.input.def -thread 8 -guide $benchmark_path/$data/$data.output5 --output $cugr2/DR_result/Ours_$data.txt --tat 2000000000 > $cugr2/DR_log/$data\_Ours_CZ.log
cd $this_path