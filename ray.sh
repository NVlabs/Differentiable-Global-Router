# run ray hyper-parameters for multiple benchmarks
# args: data, # of trials
# python test_lr.py ispd18_test5_metal5 40 4
# python test_opt.py ispd18_test5_metal5 40 4
python test_scheduler.py ispd18_test5_metal5 30 8
python test_scheduler.py ispd18_test8_metal5 30 6
