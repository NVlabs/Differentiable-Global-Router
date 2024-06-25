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
import data
from data import random_data, process_pool,routing_region
import util
# from ilp import LP_data_prep, solve_by_lp
import model
import timeit
# arguments
import argparse
from torch_scatter import scatter_max
# from ray import tune
import tracemalloc
import os
import numpy as np
import sys
# fix the seed

# torch.manual_seed(0)
# np.random.seed(0)
# torch.backends.cudnn.deterministic = True
# import random
# random.seed(0)

old_out = sys.stdout
class StAmpedOut:
    """Stamped stdout."""
    nl = True
    start = timeit.default_timer()
    def write(self, x):
        """Write function overloaded."""
        if x == '\n':
            old_out.write(x)
            self.nl = True
        elif self.nl:
            old_out.write('[%.3f] %s' % (float(timeit.default_timer() - self.start), x))
            self.nl = False
        else:
            old_out.write(x)
    def flush(var):         
        pass
    
sys.stdout = StAmpedOut()

tracemalloc.start()
parser = argparse.ArgumentParser()
# random data information, not needed if data_path is not None
parser.add_argument('--xmax', type=int, default=50)
parser.add_argument('--ymax', type=int, default=50)
parser.add_argument('--capacity', type=float, default=1.0, help = 
                    "when data_path is None, the capacity of the routing region; otherwise, the capacity ratio to scale the capacity")
parser.add_argument('--net_num', type=int, default=50)
parser.add_argument('--max_pin_num', type=int, default=3)
parser.add_argument('--net_size', type=int, default=10)

# data_path, str, the path for contest benchmark, when value is None, then, use random data
parser.add_argument('--data_path', type=str, default='/scratch/weili3/cu-gr-2/run/bsg.pt')

# DL hyperparameters
parser.add_argument('--lr', type=float, default=0.3)
parser.add_argument('--optimizer', type=str, default='rmsprop')
parser.add_argument('--scheduler', type=str, default='constant')
parser.add_argument('--t', type=float, default=1, help = "temperature scale")
parser.add_argument('--tree_t', type=float, default=0.9, help = "temperature scale for tree candidates, since we only output one tree, we use 0.85 here")
parser.add_argument('--iter', type=int, default=1000)
parser.add_argument('--epoch_iter', type=int, default=1, help = "how many iterations for each epoch")
parser.add_argument('--act', type=str, default='sigmoid', help = 'relu, leaky_relu, celu, exp, sigmoid')
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--beta1', type=float, default=0.9)
parser.add_argument('--via_coeff', type=float, default=4)
parser.add_argument('--wl_coeff', type=float, default=0.5)
parser.add_argument('--overflow_coeff', type=float, default=1, help ="500 is already included in unit_length_short_cost")
# celu_alpha, default is 2.0
parser.add_argument('--celu_alpha', type=float, default=2.0)
parser.add_argument('--act_scale', type=float, default=0.5, help="scale the cap - demand value")
parser.add_argument('--via_layer', type=float, default=1.5,help= "how many layers does a via occupy? This is calculated by sqrt(num_layer) * args.via_layer")
parser.add_argument('--use_gumble', type=bool, default=True)


# candidate pool hyperparameters
# pattern_level, default is 2
parser.add_argument('--pattern_level', type=int, default=1, help = "1 only l-shape, 2 add z shape 3 add c shape")
# z step, default is 2
parser.add_argument('--z_step', type=int, default=3, help = 
                    "If we have n possible z-shape turing points, then we pick turning point every z_step, which will generate int(n/(z_step)) z-shape routing candidates")
parser.add_argument('--max_z', type=int, default=10, help = 
                    "the maximum number of z shape routing candidates, if we have more than max_z candidates, we will increase the z_step until we have less than max_z candidates")
# c step, default is 2 
parser.add_argument('--c_step', type=int, default=3, help = 
                    "If we have n possible c-shape turing points, then we pick turning point every c_step, which will generate int(n/(c_step)) c-shape routing candidates")
parser.add_argument('--max_c', type=int, default=20, help = 
                    "the maximum number of c shape routing candidates, if we have more than max_c candidates, we will increase the c_step until we have less than max_c candidates")
parser.add_argument('--max_c_out_ratio', type=float, default=5, help = 
                    "when pick c turning points, we need to extend the edge, this is the maximum ratio that we extend the edge, default is 1, means if x width is n, we will extend n along with x-axis")
parser.add_argument('--pin_ratio', type=float, default=1, help = "Influence of pin density to the capacity")
parser.add_argument('--local_net_ratio', type=float, default=1, help = "Influence of loca_net density to the capacity")
parser.add_argument('--add_CZ', type=bool, default=False, help = 'whether add c and z shape candidates in the candidate pool, if so, will add z after 20% iterations, and add c after 50% iterations')


# framework parameters
parser.add_argument('--add_via', type=bool, default=True, help="If True, will add via as a demand in overflow cost cal")
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--select_threshold', type=float, default=1.0, help = 
                    "Set a Probability threshold: t, select candidates from lager p to smaller p, until sum of candidate probabilities are larger than t")
parser.add_argument('--read_new_tree', type=bool, default=False, help ='whether read new tree generated from phase 2 of CUGR2. If true, will read new trees, and those candidates will be used in the framework')
parser.add_argument('--tree_file_path', type=str, default=None)
# output_name, default is None
parser.add_argument('--output_name', type=str, default='output', help = "the name of the output file")
# whether use ilp_metric, default is True, if True, will use ilp_metric from DAC23 paper, which did not consider the influence of vias and pins to the congestion
parser.add_argument('--use_ilp_metric', type=bool, default=False, help = "whether use ilp_metric from DAC23 paper, which did not consider the influence of vias and pins to the congestion")

args = parser.parse_args()
args.device = 'cuda:'+str(args.device) if torch.cuda.is_available() else 'cpu'
args.data_name = args.data_path.split('/')[-1].split('.')[0]

# args.lr = args.lr / args.epoch_iter, otherwise, it is overfitting
args.lr = args.lr / args.epoch_iter

# set CUDA device as 0 
CUDA_VISIBLE_DEVICES=args.device
start = timeit.default_timer()
# if data_path is None, then we use random generated data
assert args.data_path is not None, "Random data is not used now"

if args.data_path is None:
    RouteNets = random_data(args.xmax,args.ymax,args.net_num,args.max_pin_num,(args.net_size,args.net_size),seed=0)
    RoutingRegion = routing_region(args.xmax,args.ymax,cap_mat = args.capacity,device = args.device)
else:
    results = torch.load(args.data_path)
    # args.data_path is replacing cu-gr in args.data_path by cu-gr-2

    RouteNets = results['net']
    # RouteNets = results['net'] 
    RoutingRegion = results['region']
    args.xmax = RoutingRegion.xmax
    args.ymax = RoutingRegion.ymax
    args.net_num = len(RouteNets)
    RoutingRegion3D = results["region3D"]
    num_layer = len(RoutingRegion3D.cap_mat_3D[0]) + len(RoutingRegion3D.cap_mat_3D[1])
    args.num_layer = num_layer
    args.hor_first = RoutingRegion3D.hor_first
    args.num_hor_layer = len(RoutingRegion3D.cap_mat_3D[0])
    args.num_ver_layer = len(RoutingRegion3D.cap_mat_3D[1])
    args.via_layer = float(np.sqrt(num_layer)) * args.via_layer
    RoutingRegion.cap_mat = [torch.stack(RoutingRegion3D.cap_mat_3D[0]).sum(0),torch.stack(RoutingRegion3D.cap_mat_3D[1]).sum(0)]
    RoutingRegion.cap_mat = [RoutingRegion.cap_mat[0] * args.capacity,RoutingRegion.cap_mat[1] * args.capacity]
    results['edge_length'] = [results['edge_length'][1],results['edge_length'][0]] # the hor and ver definitions are different from CUGR2, therefore, we inverse
    # get the edge demand inflence by physical pins (and steiner points if only one tree)
    hor_edge_demand,ver_edge_demand,hor_pin_demand, ver_pin_demand = util.get_pin_demand(RouteNets, args,results['layers'], results['edge_length'])
    hor_local_net_demand, ver_local_net_demand = util.get_local_net(RouteNets, args)

    RoutingRegion.cap_mat[0] = RoutingRegion.cap_mat[0] - hor_local_net_demand * args.local_net_ratio
    RoutingRegion.cap_mat[1] = RoutingRegion.cap_mat[1] - ver_local_net_demand * args.local_net_ratio
    if args.use_ilp_metric is False:
        # if not use ilp metric, we need to consider pin influence
        RoutingRegion.cap_mat[0] = RoutingRegion.cap_mat[0] - torch.tensor(hor_edge_demand) * args.pin_ratio 
        RoutingRegion.cap_mat[1] = RoutingRegion.cap_mat[1] - torch.tensor(ver_edge_demand) * args.pin_ratio
    hor_edge_length = torch.tensor(results['edge_length'][0]).to(args.device).reshape(1,-1)
    ver_edge_length = torch.tensor(results['edge_length'][1]).to(args.device).reshape(-1,1)
    RoutingRegion.to(args.device)
    hor_pin_demand = torch.tensor(hor_pin_demand).to(args.device)
    ver_pin_demand = torch.tensor(ver_pin_demand).to(args.device)
    m2_pitch = results['m2_pitch']
    min_unit_length_short_cost = min([layer['unit_length_short_cost'] for layer in results['layers']])
    util.print_data_stat(args)
    if args.read_new_tree:
        # args.data_path is removing final /....pt
        cugr2_dir_path = args.data_path[:-len(args.data_path.split('/')[-1])]
        RouteNets = util.read_new_tree(RouteNets,cugr2_dir_path,args)
    

print("args: ", args)
print("Data generation time: ", timeit.default_timer() - start)

# For each net, generate a candidate pool, that is
start = timeit.default_timer()
# if ./tmp/{data_name}_candidate_pool.pt exists, then load candidate_pool, otherwise, generate candidate_pool
# if False:
if os.path.exists('./tmp/' + args.data_name + '_candidate_pool.pt') and args.read_new_tree is False:
    candidate_pool = torch.load('./tmp/' + args.data_name + '_candidate_pool.pt')
    print("candidate pool loaded")
else:
    candidate_pool = util.get_initial_candidate_pool(RouteNets, args.xmax, args.ymax, device = args.device, edge_length = results['edge_length'],pattern_level = args.pattern_level, max_z = args.max_z, z_step = args.z_step,c_step = args.c_step, max_c= args.max_c, max_c_out_ratio = args.max_c_out_ratio)
    torch.save(candidate_pool, './tmp/' + args.data_name + '_candidate_pool.pt')
pool_generation_time = timeit.default_timer() - start
print("Initial candidate pool generation time: ", timeit.default_timer() - start)

# if ./tmp/{data_name}_p_index.pt exists, then load p_index, otherwise, generate p_index
# if False:
if os.path.exists('./tmp/' + args.data_name + '_p_index.pt') and args.read_new_tree is False:
    p_index, p_index_full, p_index2pattern_index,hor_path, ver_path, wire_length_count, via_info, tree_p_index, tree_index_per_candidate, tree_p_index2pattern_index, tree_p_index_full = torch.load('./tmp/' + args.data_name + '_p_index.pt', map_location = args.device)
else:
    p_index, p_index_full, p_index2pattern_index,hor_path, ver_path, wire_length_count, via_info, tree_p_index, tree_index_per_candidate, tree_p_index2pattern_index, tree_p_index_full= process_pool(candidate_pool,args.xmax, args.ymax,device = args.device)
    torch.save((p_index, p_index_full, p_index2pattern_index,hor_path, ver_path, wire_length_count, via_info, tree_p_index, tree_index_per_candidate, tree_p_index2pattern_index, tree_p_index_full), './tmp/' + args.data_name + '_p_index.pt')

if args.read_new_tree is False: 
    tree_p_index = None

config = {
        "lr": args.lr,
        "t": args.t,
        "tree_t": args.tree_t,
    }

    # run the experiment for 5 times and plot the 5 cost curves
overflow_cost_list = []
via_cost_list = []
best_cost = 1e10
best_continue_cost = 1e10
worst_cost = 0
p_generation_time = 0 # time to use softmax and calculate p
cost_inference_time = 0 # time to calculate the cost
back_time = 0 # time to back propagate
start = timeit.default_timer()

net = model.Net(p_index,pattern_level = args.pattern_level,device= args.device,use_gumble = args.use_gumble, tree_p_index = tree_p_index, tree_index_per_candidate = tree_index_per_candidate).to(args.device)
if args.optimizer == 'adam':
    optimizer = torch.optim.Adam(net.parameters(), weight_decay = args.weight_decay, betas =(args.beta1,0.999),lr=config["lr"])
elif args.optimizer == 'sgd':
    optimizer = torch.optim.SGD(net.parameters(), weight_decay = args.weight_decay, lr=config["lr"])
elif args.optimizer == 'rmsprop':
    optimizer = torch.optim.RMSprop(net.parameters(), weight_decay = args.weight_decay, lr=config["lr"])
elif args.optimizer == 'adagrad':
    optimizer = torch.optim.Adagrad(net.parameters(), weight_decay = args.weight_decay, lr=config["lr"])
else:
    raise NotImplementedError("Optimizer not implemented")

if args.scheduler == 'step':
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(args.iter/10), gamma=0.8)
elif args.scheduler == 'cosine':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.iter, eta_min=0.0001)
elif args.scheduler == 'linear':
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1 - epoch / args.iter)
elif args.scheduler == 'constant':
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1)
elif args.scheduler == 'restart':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=int(args.iter/10), T_mult=1, eta_min=0.0001)
else:
    raise NotImplementedError("Scheduler not implemented")
# elif args.scheduler == 'plateau':
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
cost_list = []
current_best_cost = 1e10
current_best_continue_cost = 1e10
t_ratio = 1 # temperature ratio
temperature = 1
tree_temperature = 1
print("Start training...")
# prev_argmax = torch.empty(1).to(args.device)
hor_cap = RoutingRegion.cap_mat[0].flatten()
ver_cap = RoutingRegion.cap_mat[1].flatten()
hor_total_elements = hor_cap.numel()
ver_total_elements = ver_cap.numel()
import math
hor_batch = math.ceil(hor_total_elements / args.epoch_iter)
ver_batch = math.ceil(ver_total_elements / args.epoch_iter)
for i in range(args.iter):
    # forward, update temperature for the softmax function
    # update temperature every arg.iter/5 iterations
    update_iteration = int(args.iter/10)
    if i % update_iteration == 0 and i != 0:
        temperature = temperature * config["t"]
        tree_temperature = tree_temperature * config["tree_t"]  
        # max_p, argmax = scatter_max(p,p_index_full)
        # overflow_cost, wl_cost, via_cost, max_overflow = model.discrete_objective_function(RoutingRegion, hor_path, ver_path, wire_length_count, via_info, argmax,args)
        print("iter %d, overflow cost: %.3f; via cost: %.3f; wl cost: %.3f; max_overflow: %.1f" % (i, overflow_cost.cpu().item()*args.epoch_iter, via_cost.cpu().item(), wire_length_cost.cpu().item(), max_overflow.cpu().item()))
       

    hor_shuffled_indices = torch.randperm(hor_total_elements)
    ver_shuffled_indices = torch.randperm(ver_total_elements)

    for j in range(args.epoch_iter):
        p_start = timeit.default_timer()
        p, candidate_p, tree_p = net.forward(temperature,tree_temperature)
        p_generation_time += timeit.default_timer() - p_start
        # calculate the objective function
        cost_start = timeit.default_timer()
        overflow_cost, via_cost, wire_length_cost, max_overflow, hor_overflow, ver_overflow = model.objective_function(RoutingRegion, hor_path, ver_path, wire_length_count, via_info, p,args,
                                                                                                                       hor_pin_demand,ver_pin_demand,hor_edge_length, ver_edge_length,min_unit_length_short_cost, m2_pitch,
                                                                                                                       iteration=j, hor_batch_size=hor_batch,ver_batch_size=ver_batch, hor_shuffled_indices=hor_shuffled_indices,ver_shuffled_indices=ver_shuffled_indices)

        cost_inference_time += timeit.default_timer() - cost_start
        back_start = timeit.default_timer()
        total_cost = overflow_cost*args.overflow_coeff * args.epoch_iter + wire_length_cost*args.wl_coeff + via_cost*args.via_coeff # In stochastic version, only 1/epoch_iter of the overflow is used, to balance with via cost and wirelength cost, we need to multiply by epoch_iter
        # total_cost = overflow_cost 
        total_cost.backward()
        optimizer.step()
        optimizer.zero_grad()
        back_time += timeit.default_timer() - back_start
    scheduler.step()

    # print("args.iteration: ", i, "overflow_cost: ", overflow_cost.cpu().item())
    cost_list.append(float(total_cost.detach().cpu()))
    overflow_cost_list.append(float(overflow_cost.detach().cpu()))
    via_cost_list.append(float(via_cost.detach().cpu()))

    # if add_CZ is enabled, we add Z after 20% iterations, and add Z + C after 50% iterations
    if args.add_CZ:
        if i == (4*update_iteration) or i == (8*update_iteration):
            cz_level = 2
            CZ_result = util.add_CZ(net.p, p_index_full, p_index2pattern_index, hor_path,ver_path,wire_length_count, via_info, tree_index_per_candidate, 
                                    candidate_pool, hor_overflow, ver_overflow, cz_level,args,results['edge_length'])
            if CZ_result is not None:
                p_index_full, p_index2pattern_index, hor_path, ver_path, wire_length_count, via_count, via_map, tree_index_per_candidate, new_p, candidate_pool = CZ_result
                print("iter %d: %d/%d new candidates are generated "% (i, new_p.shape[0] - net.p.shape[0], net.p.shape[0]))
                net.p = torch.nn.Parameter(new_p.float(), requires_grad = True)
                net.p_full_index = p_index_full
                if net.tree_p_index is not None:
                    net.tree_index_per_candidate = tree_index_per_candidate
                if args.optimizer == 'adam':
                    optimizer = torch.optim.Adam(net.parameters(), weight_decay = args.weight_decay, betas =(args.beta1,0.999),lr=config["lr"])
                elif args.optimizer == 'sgd':
                    optimizer = torch.optim.SGD(net.parameters(), weight_decay = args.weight_decay, lr=config["lr"])
                elif args.optimizer == 'rmsprop':
                    optimizer = torch.optim.RMSprop(net.parameters(), weight_decay = args.weight_decay, lr=config["lr"])
                elif args.optimizer == 'adagrad':
                    optimizer = torch.optim.Adagrad(net.parameters(), weight_decay = args.weight_decay, lr=config["lr"])
                else:
                    raise NotImplementedError("Optimizer not implemented")
                via_info = (via_map,via_count)

    if cost_list[-1] < current_best_continue_cost:
        current_best_continue_cost = cost_list[-1]
    if i == args.iter - 1:
        max_p, argmax = scatter_max(p,p_index_full)
        current_best_cost = (overflow_cost + wire_length_cost*args.wl_coeff + via_cost*args.via_coeff).cpu().item()
        # util.visualize_result(args.xmax,args.ymax, hor_path, ver_path, p, name=args.data_name +'_'+ str(config["t"]) + '_' + str(config["lr"]),
        #                 capacity_mat=RoutingRegion.cap_mat,
        #                 caption="cost: " + str(current_best_cost) + " temperature: " + str(temperature) + " lr: " + str(config["lr"]))
        # util.visualize_p(max_p.cpu().detach().numpy(),"s1_p_" + args.data_name)

train_time = (timeit.default_timer() - start)
            
best_cost = current_best_cost
best_continue_cost = current_best_continue_cost
worst_cost = current_best_cost

# print final total cost, overflow cost, via cost
# Set a Probability threshold: t, select candidates from lager p to smaller p, until sum of candidate probabilities are larger than t
print("best cost: ", best_cost, "overflow cost: ", float(overflow_cost.cpu().detach()), "via cost: ", via_cost_list[-1])

if args.read_new_tree is True:
    selected_tree_full_index = util.write_tree_result(RouteNets,tree_p, tree_p_index_full, tree_p_index2pattern_index,args.data_name,write_tree = args.read_new_tree)
    # only pick 2-pin candidates from those picked trees
    # selected_p_full_index = torch.repeat_interleave(selected_tree_full_index, tree_index_per_candidate)
    selected_p_full_index = selected_tree_full_index[tree_index_per_candidate]
    # set not selected p_index2pattern_index as 0
    p_index2pattern_index[(selected_p_full_index == False).cpu()] = 0

util.write_CUGR_input(RouteNets,candidate_p,p_index_full,candidate_pool,p_index2pattern_index,args.data_name + '_' + args.output_name, args.select_threshold)


# plot cost_list and save in ./figs/cost_{data_name}.png
# import matplotlib.pyplot as plt
# import seaborn as sns
# sns.set()
# plt.figure()
# plt.plot(cost_list)
# plt.plot(overflow_cost_list)
# plt.plot(via_cost_list)
# plt.legend(["total_cost","overflow_cost","via_cost"])
# plt.xlabel("iteration")
# plt.ylabel("cost")
# plt.savefig("./figs/s1_cost_" + args.data_name + ".png")
# plt.close()


# visualization

# util.save_path_result(RouteNets,argmax,candidate_pool,p_index2pattern_index,args.data_name)

# # ILP
# if len(p) > 9999999900000:
#     hor_cap, ver_cap, hor_path, ver_path = LP_data_prep(RoutingRegion,hor_path,ver_path)
#     start = timeit.default_timer()
#     ilp_p, ilp_obj = solve_by_lp(hor_cap, ver_cap, hor_path, ver_path,p_index,pow = args.use_pow)
#     ilp_time = timeit.default_timer() - start
#     lp_p, lp_obj = solve_by_lp(hor_cap, ver_cap, hor_path, ver_path,p_index, is_ilp = False,pow = args.use_pow)
# else:
ilp_obj = 1e10
lp_obj = 1e10
ilp_time = 0
    


# print("ILP/LP overflow cost: ", ilp_obj,lp_obj)
# print("Gradient descent overflow cost: ", best_cost)   

# save args.xmax, args.ymax, args.capacity, args.net_num, args.max_pin_num, args.net_size, pool_generation_time, train_time, ilp_obj, best_cost, worst_cost to a csv file

import csv
import os
# create csv file if not exist
if not os.path.exists('./step1.csv'):
    with open('./step1.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # row_list is a list of strings of args
        row_list = []
        for arg in vars(args):
            row_list.append(arg)
        row_list += ['num_candidates','pool_generation_time', 'train_time', 'ilp_time','ilp_obj', 'best_cost','overflow_cost', 'via_cost', 'wire_length_cost','max_overflow', 'worst_cost', 'lp_obj', 'best_continue_cost', 'p_generation_time', 'cost_inference_time', 'back_time', 'peak_gpu_memory (MB)','peak_cpu_memory (MB)']
        writer.writerow(row_list)
# write to csv file 
torch.cuda.empty_cache()
with open('./step1.csv', 'a', newline='') as csvfile:
    writer = csv.writer(csvfile)
    # write each args
    row_list = []
    for arg in vars(args):
        row_list.append(getattr(args, arg))
    row_list += [p_index[-1],pool_generation_time, train_time,ilp_time, ilp_obj, best_cost,overflow_cost.cpu().item(), via_cost.cpu().item(), wire_length_cost.cpu().item(),max_overflow.cpu().item(), worst_cost, lp_obj, best_continue_cost, p_generation_time, cost_inference_time, back_time,torch.cuda.max_memory_allocated()/(1024*1024),tracemalloc.get_traced_memory()[1]/(1024*1024)]
    writer.writerow(row_list)
