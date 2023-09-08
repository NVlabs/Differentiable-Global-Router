import os
import re
import ray
import subprocess
from ray import tune
import os
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
import torch
import numpy
import sys
# set data as the input argment
data = sys.argv[1]
# set available GPU id as sys.argv[2]
os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[2]
NUM_MODELS = 5


def train_model(config):
    # Variables

    benchmark_path = "/scratch/weili3/cu-gr-2/benchmark"
    cugr2 = "/scratch/weili3/cu-gr-2"
    this_path = "/home/weili3/Differentiable-Global-Router"
    os.chdir(this_path)
    # Execute python3 command
    subprocess.run(["python", "main.py", "--data_path", os.path.join(cugr2, "run", f"{data}.pt"), "--lr", str(config['learning_rate']), "--t", str(config['temperature']), "--via_coeff", str(config['via_coeff']), 
                   "--pin_ratio", str(config['pin_ratio']), "--select_threshold", str(config['select_threshold'])], stdout=open(os.path.join(cugr2, "GR_log", f"{data}_Ours.log"), "w"))

    # Change directory and execute route command
    os.chdir(os.path.join(cugr2, "run"))
    subprocess.run([
        "./route",
        "-lef", os.path.join(benchmark_path, data, f"{data}.input.lef"),
        "-def", os.path.join(benchmark_path, data, f"{data}.input.def"),
        "-output", os.path.join(benchmark_path, data, f"{data}.output3"),
        "-dgr", os.path.join(this_path, "CUGR2_guide", f"CUgr_{data}_0_0.txt")
    ], stdout=open(os.path.join(cugr2, "GR_log", f"{data}_Ours.log"), "w"))

    # # Change directory and execute drcu command
    # os.chdir(cugr2)
    # subprocess.run([
    #     "./drcu",
    #     "-lef", os.path.join(benchmark_path, data, f"{data}.input.lef"),
    #     "-def", os.path.join(benchmark_path, data, f"{data}.input.def"),
    #     "-thread", "8",
    #     "-guide", os.path.join(benchmark_path, data, f"{data}.output3"),
    #     "--output", os.path.join(cugr2, "DR_result", f"Ours_{data}.txt"),
    #     "--tat", "2000000000"
    # ], stdout=open(os.path.join(cugr2, "DR_log", f"{data}_Ours.log"), "w"))

    # Change back to the original directory
    os.chdir(this_path)

    # Parse the output log file
    with open(f"{cugr2}/GR_log/{data}_Ours.log", "r") as f:
        content = f.read()
        wire_length = int(re.search(r"wire length \(metric\):\s+(\d+)", content).group(1))
        via_count = int(re.search(r"total via count:\s+(\d+)", content).group(1))
        overflow = int(re.search(r"total wire overflow:\s+(\d+)", content).group(1))
        min_resource = float(re.search(r"min resource:\s+(-?\d+(\.\d+)?)", content).group(1))

    # Compute the score
    score = wire_length * 0.5 + via_count * 2 + overflow * 3000 - min_resource * 10000
    # score = via_count * 2 + overflow * 3000 - min_resource * 10000

    # Report the metrics
    return {"score":score, "wire_length":wire_length, "via_count":via_count, "overflow":overflow, "min_resource":min_resource}


trainable_with_resources = tune.with_resources(train_model,
resources=lambda spec: {"gpu": 1} if torch.cuda.is_available() else {"gpu": 0})

config = {
    "learning_rate": tune.loguniform(1e-4, 1),
    "temperature": tune.choice([0.8, 0.85, 0.9, 0.95, 1]),
    "via_coeff": tune.loguniform(1e-4, 1e2),
    "pin_ratio": tune.uniform(0.5,1.5),
    "select_threshold": tune.choice([0.8, 0.85, 0.9, 0.95, 1])
    }


tuner = tune.Tuner(
    trainable_with_resources,    
    tune_config=tune.TuneConfig(
        num_samples=int(sys.argv[3]),
        max_concurrent_trials=1
    ),
    param_space=config
)


analysis = tuner.fit()
results_df = analysis.get_dataframe()
# save df to a csv (append if the file exists)
results_df.to_csv(f"./ray_results/{data}.csv", mode='a', header=False)