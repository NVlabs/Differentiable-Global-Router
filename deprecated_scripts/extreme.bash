# xmax,ymax in [20,50,100,1000,10000]
# capacity in [1,2,3,5,10]
# net_num in [5,10,1000,10000,100000]
# net_size is xmax/5
for pattern_level in 1
do
    for xmax in 50
    do
        # ymax is the same as xmax
        ymax=$xmax
        for capacity in 2
        do
            for net_num in 100
            do
                echo "xmax: $xmax, ymax: $ymax, capacity: $capacity, net_num: $net_num, net_size: 10, pattern_level: $pattern_level"
                python main.py --xmax $xmax --ymax $ymax --capacity $capacity --net_num $net_num --net_size 10 --pattern_level $pattern_level --num_experiment 100
            done
        done
    done
done