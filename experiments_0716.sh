# run the whole framework
# data_list is [ispd18_test1, ispd18_test2, ispd18_test5]
data_list=("ispd18_test1" "ispd18_test2" "ispd18_test3" "ispd19_test1" "ispd19_test2" "ispd19_test3")
# data_list=("ispd18_test2")
s2_via=("0.1" "0.01" "0.001" "0.0001")
s1_via=("0.02" "0.002" "0.0002" "0.00002")
s1_wl=("0.1" "0.01" "0.001")


for s1wl in "${s1_wl[@]}"
do
    for s1via in "${s1_via[@]}"
    do
        for data in "${data_list[@]}"
        do
            # run the whole framework
            echo "data: $data"
            # python3 main.py --data_path /home/scratch.rliang_hardware/wli1/cu-gr/run/$data.pt --pattern_level 3 --capacity 0.5 --via_coeff $s1via --wl_coeff $s1wl
            python3 layer_assignment.py  --data_path /home/scratch.rliang_hardware/wli1/cu-gr/run/$data.pt  --capacity 1
        done
    done
done


# for s2via in "${s2_via[@]}"
# do
#     for data in "${data_list[@]}"
#     do
#         # run the whole framework
#         echo "data: $data"
#         python3 main.py --data_path /home/scratch.rliang_hardware/wli1/cu-gr/run/$data.pt --pattern_level 3 --capacity 0.5
#         python3 layer_assignment.py  --data_path /home/scratch.rliang_hardware/wli1/cu-gr/run/$data.pt  --capacity 1  --via_coeff $s2via
#     done
# done