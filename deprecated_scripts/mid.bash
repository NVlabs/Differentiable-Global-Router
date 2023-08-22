# xmax,ymax in [20,50,100,1000,10000]
# capacity in [1,2,3,5,10]
# net_num in [5,10,1000,10000,100000]
# net_size is xmax/5
for pattern_level in 1 2
do
    for xmax in 1000
    do
        # ymax is the same as xmax
        ymax=$xmax
        for capacity in 1 2 3 5 10
        do
            for net_num in 5 10 20 50 100 1000 10000 100000
            do
                net_size=$((xmax/5))
                # skip if net_num > xmax * xmax * capacity / 2 or net_num < xmax/5
                if [ $net_num -gt $((xmax * xmax * capacity / 2)) ] || [ $net_num -lt $((xmax * xmax * capacity / 20)) ]
                then
                    continue
                fi
                echo "xmax: $xmax, ymax: $ymax, capacity: $capacity, net_num: $net_num, net_size: $net_size, pattern_level: $pattern_level"
                python main.py --xmax $xmax --ymax $ymax --capacity $capacity --net_num $net_num --net_size $net_size --pattern_level $pattern_level
            done
        done
    done
done