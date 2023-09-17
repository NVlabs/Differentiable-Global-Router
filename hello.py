import subprocess
import os
config = {
    # "learning_rate": tune.loguniform(1e-5, 1),
    # "temperature": tune.choice([0.8, 0.85, 0.9, 0.95, 1]),
    "learning_rate":1e-3,
    "temperature": 0.85,
    "overflow_coeff": 1e-4,
    # "select_threshold": tune.choice([0.8, 0.85, 0.9, 0.95, 1]),
    "act_scale": 1e-2,
    "act":'leaky_relu'
    }
task_id = 520
data = 'ispd18_test2'
benchmark_path = "/scratch/weili3/cu-gr-2/benchmark"
cugr2 = "/scratch/weili3/cu-gr-2"
this_path = "/home/weili3/Differentiable-Global-Router"
subprocess.run(["python", "main_stochastic.py", "--data_path", os.path.join(cugr2, "run", f"{data}.pt"), "--lr", str(config['learning_rate']), "--t", str(config['temperature']), "--overflow_coeff", str(config['overflow_coeff']), 
                "--act_scale", str(config['act_scale']), "--act", str(config['act']),"--epoch_iter", "1" , "--output_name", str(task_id)])