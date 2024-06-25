# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""
The main script for the back-propagation algorithm
"""

import torch
from data import random_data, get_path_tensor,routing_region
from util import get_initial_candidate_pool, visualize_result
from ilp import LP_data_prep, solve_by_lp
import model
import timeit
# arguments
import argparse
from torch_scatter import scatter_max
from ray import tune
from ray.air import Checkpoint, session
from ray.tune.schedulers import ASHAScheduler

parser = argparse.ArgumentParser()
# random data information
parser.add_argument('--xmax', type=int, default=50)
parser.add_argument('--ymax', type=int, default=50)
parser.add_argument('--capacity', type=int, default=1)
parser.add_argument('--net_num', type=int, default=50)
parser.add_argument('--max_pin_num', type=int, default=3)
parser.add_argument('--net_size', type=int, default=10)

# data_path, str, the path for contest benchmark, when value is None, then, use random data
parser.add_argument('--data_path', type=str, default='$cugr2/run/ispd18_test1.pt')
# data_name, str, default is ispd18_test1
# parser.add_argument('--data_name', type=str, default='ispd18_test1')

# DL hyperparameters
parser.add_argument('--lr', type=float, default=0.2)
parser.add_argument('--iter', type=int, default=1000)
parser.add_argument('--num_experiment', type=int, default=1)

# candidate pool hyperparameters
# pattern_level, default is 2
parser.add_argument('--pattern_level', type=int, default=2, help = "1 only l-shape, 2 add z shape 3 add monotonic shape")
# z step, default is 4
parser.add_argument('--z_step', type=int, default=4, help = 
                    "If we have n possible z-shape turing points, then we pick turning point every z_step, which will generate int(n/(z_step)) z-shape routing candidates")
parser.add_argument('--max_z', type=int, default=10, help = 
                    "the maximum number of z shape routing candidates, if we have more than max_z candidates, we will increase the z_step until we have less than max_z candidates")

parser.add_argument('--device', type=str, default='cuda:0')
args = parser.parse_args()
args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# set CUDA device as 0 
CUDA_VISIBLE_DEVICES=0
start = timeit.default_timer()
# if data_path is None, then we use random generated data
if args.data_path is None:
    RandomData = random_data(args.xmax,args.ymax,args.net_num,args.max_pin_num,(args.net_size,args.net_size),seed=0)
    RoutingRegion = routing_region(args.xmax,args.ymax,cap_mat = args.capacity,device = args.device)
else:
    results = torch.load(args.data_path)
    RandomData = results['net']
    RoutingRegion = results['region']
    RoutingRegion.to(args.device)
    args.xmax = RoutingRegion.xmax
    args.ymax = RoutingRegion.ymax
args.data_name = args.data_path.split('/')[-1].split('.')[0]
print("Data generation time: ", timeit.default_timer() - start)

# For each net, generate a candidate pool, that is
start = timeit.default_timer()
candidate_pool,p_index, p_index2pattern_index, p_index_full = get_initial_candidate_pool(RandomData, args.xmax, args.ymax,
                                                                                         device = args.device, pattern_level = args.pattern_level, max_z = args.max_z, z_step = args.z_step)
pool_generation_time = timeit.default_timer() - start
print("Initial candidate pool generation time: ", timeit.default_timer() - start)

hor_path, ver_path = get_path_tensor(candidate_pool,p_index, p_index2pattern_index,device = args.device)


config = {
        "lr": tune.loguniform(1e-3, 10),
        # temperature is from 0.8, 0.9, 0.95, 0.99
        "temperature": tune.choice([0.8, 0.85, 0.9]),
    }

def train(config):
    # run the experiment for 5 times and plot the 5 cost curves
    cost_lists = []
    best_cost = 1e10
    best_continue_cost = 1e10
    worst_cost = 0
    start = timeit.default_timer()
    for experiment in range(args.num_experiment):
        net = model.Net(p_index,pattern_level = args.pattern_level,device= args.device).to(args.device)
        # initialize the trainable torch parameters with length  = p_index[-1]
        args.iteration =  10
        optimizer = torch.optim.Adam(net.parameters(), lr=config["lr"])
        cost_list = []
        current_best_cost = 1e10
        current_best_continue_cost = 1e10
        temperature = 1
        for i in range(args.iter):
            # forward, update temperature for the softmax function
            if i % 100 == 0:
                temperature = temperature * config["temperature"]
            p = net.forward(temperature)
            # calculate the objective function
            overflow_cost = model.objective_function(RoutingRegion, hor_path, ver_path, p)
            _, argmax = scatter_max(p,p_index_full)
            discrete_cost = model.discrete_objective_function(RoutingRegion, hor_path, ver_path, argmax)
            # visualize_result(RandomData,hor_path,ver_path,p, name = 'args.iter_' + str(i))
            # backward
            overflow_cost.backward()
            optimizer.step()
            optimizer.zero_grad()
            # print("args.iteration: ", i, "overflow_cost: ", overflow_cost.cpu().item())
            cost_list.append(float(overflow_cost.detach().cpu()))
            if cost_list[-1] < current_best_continue_cost:
                current_best_continue_cost = cost_list[-1]
            if discrete_cost.cpu().item() < current_best_cost:
                current_best_cost = discrete_cost.cpu().item()
                visualize_result(args.xmax,args.ymax, hor_path, ver_path, p, name=str(config["temperature"]) + '_' + str(config["lr"]),
                                capacity_mat=RoutingRegion.cap_mat,
                                caption="cost: " + str(current_best_cost) + " temperature: " + str(temperature) + " lr: " + str(config["lr"]))
        if current_best_cost < best_cost:
            best_cost = current_best_cost
        if current_best_continue_cost < best_continue_cost:
            best_continue_cost = current_best_continue_cost
        if current_best_cost > worst_cost:
            worst_cost = current_best_cost
        cost_lists.append(cost_list)
        session.report({"best_cost":discrete_cost.cpu().item()})
trainable_with_resources = tune.with_resources(train,
# resources={"gpu": 0})
resources=lambda spec: {"gpu": 1} if torch.cuda.is_available() else {"gpu": 0})

tuner = tune.Tuner(
    trainable_with_resources,    
    tune_config=tune.TuneConfig(
        num_samples=10
    ),
    param_space=config
)

analysis = tuner.fit()
results_df = analysis.get_dataframe()
# save df to a csv
results_df.to_csv("results_df.csv")

