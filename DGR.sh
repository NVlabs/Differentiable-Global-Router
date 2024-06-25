# run different parameters for result analysis
data="${1:-ispd18_test1}"
benchmark_path=$2
cugr2=$3
this_path=$4
method="Ours"
echo "Processing Our First Round... data: $data"
<<<<<<< HEAD:DGR.sh

# Run 
python3 main.py --data_path $cugr2/run/$data.pt --select_threshold 1 --t 1
# python3 main_stochastic.py --data_path $cugr2/run/$data.pt --lr 0.0151052486678429 --pin_ratio 1.1524306252869 --select_threshold 1 --t 1 --via_coeff 0.000469797384581491
=======
# python3 main_stochastic.py --data_path $cugr2/run/$data.pt --output_name $method --select_threshold 0.9 --device 2
>>>>>>> main:ours.sh
cd $cugr2/run/
./route -lef $benchmark_path/$data/$data.input.lef -def $benchmark_path/$data/$data.input.def -sort 0 -output $benchmark_path/$data/$data.$method.guide -dgr $this_path/CUGR2_guide/CUgr_$data\_$method.txt > $cugr2/GR_log/$data\_$method.log
cd $cugr2
<<<<<<< HEAD:DGR.sh
./drcu -lef $benchmark_path/$data/$data.input.lef -def $benchmark_path/$data/$data.input.def -thread 8 -guide $benchmark_path/$data/$data.output3 --output $cugr2/DR_result/Ours_$data.txt --tat 2000000000 > $cugr2/DR_log/$data\_Ours.log
cd $this_path
=======
./drcu -lef $benchmark_path/$data/$data.input.lef -def $benchmark_path/$data/$data.input.def -thread 32 -guide $benchmark_path/$data/$data.$method.guide --output $cugr2/DR_result/${method}_$data.txt --tat 2000000000 > $cugr2/DR_log/$data\_$method.log
cd $this_path
sh run_innovus.sh $data $method
>>>>>>> main:ours.sh
