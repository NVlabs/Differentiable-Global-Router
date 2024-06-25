# run different parameters for result analysis
data="${1:-ispd18_test5_metal5}"
benchmark_path=$2
cugr2=$3
this_path=$4
method="CZ"
echo "Processing Ours w. CZ Round... data: $data"
python3 main_stochastic.py --data_path $cugr2/run/$data.pt --output_name $method --add_CZ True --epoch_iter 1 
cd $cugr2/run/
./route -lef $benchmark_path/$data/$data.input.lef -def $benchmark_path/$data/$data.input.def -output $benchmark_path/$data/$data.$method.guide -dgr $this_path/CUGR2_guide/CUgr_$data\_$method.txt > $cugr2/GR_log/$data\_$method.log
cd $cugr2
./drcu -lef $benchmark_path/$data/$data.input.lef -def $benchmark_path/$data/$data.input.def -thread 32 -guide $benchmark_path/$data/$data.$method.guide --output $cugr2/DR_result/${method}_$data.txt --tat 2000000000 > $cugr2/DR_log/$data\_$method.log
cd $this_path
sh run_innovus.sh $data $method