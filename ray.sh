# run ray hyper-parameters for multiple benchmarks
# args: data, gpu, # of trials
# python test.py ispd18_test1 0 300 &
# python test.py ispd18_test5 1 100 &
python test.py ispd18_test5_metal5 160 16
python test.py ispd18_test8_metal5 80 8 
# python test.py ispd18_test10_metal5 4 30 &
python test.py ispd19_test7_metal5 40 4
# python test.py ispd19_test8_metal5 6 30 &
# python test.py ispd19_test9_metal5 7 30 &
