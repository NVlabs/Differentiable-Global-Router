import os
import re
import ray
import subprocess
from ray import tune
import os
import sys
os.environ["MKL_THREADING_LAYER"] = "GNU"
# set data as the input argment
data = sys.argv[1]

def train_model(config):
    task_id = ray.get_runtime_context().get_task_id()

    benchmark_path = "/scratch/weili3/cu-gr-2/benchmark"
    cugr2 = "/scratch/weili3/cu-gr-2"
    this_path = "/home/weili3/Differentiable-Global-Router"
    os.chdir(this_path)
    # Execute python3 command
    subprocess.run(["python", "main_stochastic.py", "--data_path", os.path.join(cugr2, "run", f"{data}.pt"), "--lr", str(config['learning_rate']), "--t", str(config['temperature']), "--overflow_coeff", str(config['overflow_coeff']), 
                    "--weight_decay", str(config['weight_decay']), "--beta1", str(config['beta1']),"--weight_decay", "1" , "--output_name", str(task_id)])
    # subprocess.run(["python", "main_stochastic.py","--output_name", str(task_id)])

    # Change directory and execute route command
    os.chdir(os.path.join(cugr2, "run"))
    subprocess.run([
        "./route",
        "-lef", os.path.join(benchmark_path, data, f"{data}.input.lef"),
        "-def", os.path.join(benchmark_path, data, f"{data}.input.def"),
        "-output", os.path.join(benchmark_path, data, f"{data}.Ours_{task_id}"),
        "-dgr", os.path.join(this_path, "CUGR2_guide", f"CUgr_{data}_{task_id}.txt")
    ], stdout=open(os.path.join(cugr2, "GR_log", f"{data}_Ours_{task_id}.log"), "w"))

    # Change directory and execute drcu command
    # os.chdir(cugr2)
    # subprocess.run([
    #     "./drcu",
    #     "-lef", os.path.join(benchmark_path, data, f"{data}.input.lef"),
    #     "-def", os.path.join(benchmark_path, data, f"{data}.input.def"),
    #     "-thread", "2",
    #     "-guide", os.path.join(benchmark_path, data, f"{data}.Ours_{task_id}"),
    #     "--output", os.path.join(cugr2, "DR_result", f"Ours_{data}.txt"),
    #     "--tat", "2000000000"
    # ], stdout=open(os.path.join(cugr2, "DR_log", f"{data}_Ours_{task_id}.log"), "w"))
    # remove os.path.join(this_path, "CUGR2_guide", f"CUgr_{data}_{task_id}.txt"
    os.remove(os.path.join(this_path, "CUGR2_guide", f"CUgr_{data}_{task_id}.txt"))
    # remove os.path.join(benchmark_path, data, f"{data}.Ours_{task_id}"),
    os.remove(os.path.join(benchmark_path, data, f"{data}.Ours_{task_id}"))
    # remove os.path.join(cugr2, "GR_log", f"{data}_Ours_{task_id}.log")
    # remove os.path.join(cugr2, "DR_log", f"{data}_Ours_{task_id}.log")
    # os.remove(os.path.join(cugr2, "DR_log", f"{data}_Ours_{task_id}.log"))


    # Change back to the original directory
    os.chdir(this_path)

    # Parse the output log file
    with open(f"{cugr2}/GR_log/{data}_Ours_{task_id}.log", "r") as f:
        content = f.read()
        wire_length = int(re.search(r"wire length \(metric\):\s+(\d+)", content).group(1))
        via_count = int(re.search(r"total via count:\s+(\d+)", content).group(1))
        overflows = re.findall(r'(\d+) / \d+ nets have overflows.', content)
        first_overflow = float(overflows[0])
        overflow = float(overflows[-1])
        min_resource = float(re.search(r"min resource:\s+(-?\d+(\.\d+)?)", content).group(1))

    # Compute the score
    score = wire_length * 0.5 + via_count * 4 + overflow * 1000 + first_overflow * 10 - min_resource * 10000
    # score = via_count * 2 + overflow * 3000 - min_resource * 10000

    os.remove(os.path.join(cugr2, "GR_log", f"{data}_Ours_{task_id}.log"))
    # Report the test
    # tune.report(score=score, wire_length=wire_length, via_count=via_count, overflow=overflow, min_resource=min_resource)
    return {"score":score, "wire_length":wire_length, "via_count":via_count, "overflow":overflow, "min_resource":min_resource, "first_overflow":    first_overflow}

ray.init(ignore_reinit_error=True, num_cpus=int(sys.argv[3]))
print("ray intialized")
trainable_with_resources = tune.with_resources(train_model,
resources=lambda spec: {"gpu": 0.5,"cpu": 1})

config = {
    # "learning_rate": tune.loguniform(1e-5, 1),
    # "temperature": tune.choice([0.8, 0.85, 0.9, 0.95, 1]),
    "learning_rate":tune.loguniform(1e-2, 1e1),
    "temperature": 0.9,
    "overflow_coeff": 1,
    # "select_threshold": tune.choice([0.8, 0.85, 0.9, 0.95, 1]),
    "weight_decay": tune.choice([0,1e-5,1e-3,1e-2]),
    "beta1":tune.uniform(0.2, 0.998),
    }


tuner = tune.Tuner(
    trainable_with_resources,    
    tune_config=tune.TuneConfig(
        num_samples=int(sys.argv[2])
    ),
    param_space=config
)


analysis = tuner.fit()
results_df = analysis.get_dataframe()
# save df to a csv (append if the file exists)
results_df.to_csv(f"./ray_results/{data}.csv", mode='a', header=True)