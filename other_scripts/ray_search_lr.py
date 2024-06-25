"""
We use ray to fine tune the parameters
"""

import os
import re
import ray
import subprocess
# ray.init(num_gpus=6)
import torch
from ray import tune
from ray.air import session
def objective_function(config):
    # Variables
    # data = "ispd18_test5_metal5"
    data = "ispd19_test8_metal5"

    benchmark_path = "/scratch/weili3/cu-gr-2/benchmark"
    cugr2 = "/scratch/weili3/cu-gr-2"
    this_path = "/home/weili3/Differentiable-Global-Router"

    # Execute python3 command
    # subprocess.run(["python", "main_stochastic.py", "--data_path", os.path.join(cugr2, "run", f"{data}.pt"), "--lr", config['learning_rate'], "--t", config['temperature'], "--via_coeff", config['via_coeff'], 
    #                "--pin_ratio", config['pin_ratio'], "--select_threshold", config['select_threshold']], stdout=open(os.path.join(cugr2, "GR_log", f"{data}_Ours.log"), "w"))
    subprocess.run(["python", "main_stochastic.py", "--data_path", os.path.join(cugr2, "run", f"{data}.pt"), "--via_layer", config['via_layer'],
                     "--select_threshold", config['select_threshold']], stdout=open(os.path.join(cugr2, "GR_log", f"{data}_Ours.log"), "w"))

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

    # Report the metrics
    return {"score":score, "wire_length":wire_length, "via_count":via_count, "overflow":overflow, "min_resource":min_resource}

# mkdir ray_results
if not os.path.exists("./ray_results"):
    os.mkdir("./ray_results")
# Start the Ray Tune experiment

trainable_with_resources = tune.with_resources(objective_function,
resources=lambda spec: {"gpu": 1} if torch.cuda.is_available() else {"gpu": 0})

# Define the search space
search_space = {
    "via_layer": tune.choice([0.5, 1, 1.5, 2, 3]),
    "select_threshold": tune.choice([0.8, 0.85, 0.9, 0.95, 1])
}

tuner = tune.Tuner(
    trainable_with_resources,    
    tune_config=tune.TuneConfig(
        num_samples=10,  # Number of trials
        max_concurrent_trials=5  # Maximum number of trials to run concurrently
    ),
    param_space=search_space
)

analysis = tuner.fit()
results_df = analysis.get_dataframe()
# save df to a csv
results_df.to_csv("results_df.csv")
# Print the best hyperparameters
print("Best hyperparameters found were: ", analysis.best_config)