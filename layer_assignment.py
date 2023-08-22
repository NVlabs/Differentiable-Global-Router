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
Given the calculated path tensor, we implement a differentiable layer assignment algorithm to assign the layer to each path.
Require:
    net_start_idx: netname to int, where the number is the index of the end edge in edge_list for each 2-pin net
    edge_list, (2,E), the edge index, edge_list[0,i] is the x index of the i_th edge, edge_list[1,i] is the y index of the i_th edge
    is_hor_list, boolean array of shape (E), True if the edge is horizontal.
    end_index (P), the index of the end edge in edge_list for each 2-pin net
    start_idx (P), the index of the start edge in edge_list for each 2-pin net
    edge_idx1 (E') stores the edge index in edge_list for the head 
    edge_idx2 (E') stores the edge index in edge_list for the tail
    weights (E') (the weight for via cost, heuritic method)
    is_steiner (P), true if the pin is a steiner pin
    scatter_index (E): scatter_index[i] = j means the i_th edge is connected to j_th pin. if the edge is an intermediate edge, then j = num_pins (a dummy pin)), 
    pin_max, pin_min, (P), max/pin layer inside each Pin object. 
"""

import torch
import model
import timeit
import util
import numpy as np
import argparse
import tracemalloc
tracemalloc.start()

parser = argparse.ArgumentParser()
# args.data_path, str, the path for contest benchmark, when value is None, then, use random data
parser.add_argument('--data_path', type=str, default='/home/scratch.rliang_hardware/wli1/cu-gr/run/ispd18_test2.pt')

parser.add_argument('-y_step_size', nargs="+", type=int,default=[5700,6840])
parser.add_argument('-y_step_count', nargs="+", type=int,default=[66,1])
parser.add_argument('-x_step_size', nargs="+", type=int,default=[6000,6800])
parser.add_argument('-x_step_count', nargs="+", type=int,default=[64,1])

# DL hyperparameters
parser.add_argument('--lr', type=float, default=0.3)
parser.add_argument('--iter', type=int, default=100)
parser.add_argument('--use_pow', type=bool, default=False)
parser.add_argument('--act', type=str, default='celu', help = 'relu, leaky_relu, celu, exp')
parser.add_argument('--via_coeff', type=float, default=0.0001)

# framework parameters
parser.add_argument('--capacity', type=float, default=1, help = 
                    "when data_path is None, the capacity of the routing region; otherwise, the capacity ratio to scale the capacity")
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--segment', type=bool, default=True, help = 
                    "if True, then use segment-based layer assignment; otherwise, do per-edge layer assignment")


args = parser.parse_args()

device = 'cuda:0' if torch.cuda.is_available() and args.device == 'cuda:0' else 'cpu'
# torch random seed
torch.manual_seed(0)
# set CUDA device as 0 
CUDA_VISIBLE_DEVICES=0
data_name = args.data_path.split("/")[-1].split(".")[0]
args.y_step_size, args.y_step_count, args.x_step_size, args.x_step_count = util.get_ispd_size(data_name)

x_min, x_max, y_min, y_max = util.step2coordinates(args.x_step_size, args.x_step_count, args.y_step_size, args.y_step_count) # x_min, x_max, y_min, y_max are arrays to store the coordinates for each grid
_, net_start_idx, net_end_idx, edge_list, is_hor_list, end_index, start_idx, edge_idx1, edge_idx2, weights, is_steiner, scatter_index, pin_max, pin_min= torch.load("./2D_result/%s.pt"%data_name)

edge_list = torch.tensor(edge_list).to(device)
hor_mask = torch.tensor(is_hor_list).to(device)
end_index = torch.tensor(end_index).to(device)
start_idx = torch.tensor(start_idx).to(device)
edge_idx1 = torch.tensor(edge_idx1).to(device)
edge_idx2 = torch.tensor(edge_idx2).to(device)
via_indicator3 = torch.stack([edge_idx1, edge_idx2], dim=0).to(device)
weights = torch.tensor(weights).to(device)
is_steiner = torch.tensor(is_steiner).to(device)
scatter_index = torch.tensor(scatter_index).to(device)
pin_max = torch.tensor(pin_max).to(device)
pin_min = torch.tensor(pin_min).to(device)

benchmark = torch.load(args.data_path)

RoutingRegion3D = benchmark["region3D"]
xmax = RoutingRegion3D.xmax
ymax = RoutingRegion3D.ymax
hor_E = hor_mask.sum()
ver_E = (hor_mask == False).sum()
hor_L = len(RoutingRegion3D.cap_mat_3D[0])
ver_L = len(RoutingRegion3D.cap_mat_3D[1])
# remove the first layer
if RoutingRegion3D.hor_first:
    hor_L -= 1
    RoutingRegion3D.cap_mat_3D = (RoutingRegion3D.cap_mat_3D[0][1:],RoutingRegion3D.cap_mat_3D[1])
else:
    ver_L -= 1
    RoutingRegion3D.cap_mat_3D = (RoutingRegion3D.cap_mat_3D[0],RoutingRegion3D.cap_mat_3D[1][1:])

full_end_index = torch.zeros(hor_mask.shape[0],dtype=torch.bool).to(device) # full_end_index[i] = True if i_th edge is the end edge
full_end_index[end_index] = True

if args.segment is True:
    # C_tensor is to count the size of each consecutive blocks (segments).
    # For example, hor_mask = [ True,True, False, False,True,True, False, False], then C = [2,2,2,2], hor_indicator = [True, False, True, False]
    # if hor_mask = [True,True, False, False,  False, False], then C = [2,4] because the number of first consecutive elements are 2, the number of second consecutive elements is 4
    change_indices = torch.nonzero(hor_mask[1:] != hor_mask[:-1]).squeeze().to(hor_mask.device)
    hor_indicator = torch.cat((change_indices, torch.tensor([len(hor_mask)-1]).to(hor_mask.device)), dim=0)
    C_tensor = torch.diff(hor_indicator + 1, prepend=torch.tensor([0]).to(hor_mask.device))
    segment_hor_mask = hor_mask[hor_indicator]
    hor_count = C_tensor[segment_hor_mask]
    ver_count = C_tensor[segment_hor_mask == False]
    net = model.LayerAssignmentNet(hor_E, hor_L, ver_E, ver_L,segment_hor_mask= segment_hor_mask, hor_count = hor_count, ver_count = ver_count).to(device)
else:
    net = model.LayerAssignmentNet(hor_E, hor_L, ver_E, ver_L).to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

# via_cost_label = util.get_via_cost_label(hor_L, ver_L, RoutingRegion3D.hor_first,device)
if RoutingRegion3D.hor_first:
    assert(hor_L >= ver_L)

cap = (torch.stack(RoutingRegion3D.cap_mat_3D[0],dim=1).to(device).reshape(xmax, ymax-1,hor_L), torch.stack(RoutingRegion3D.cap_mat_3D[1],dim=2).to(device).reshape(xmax-1,ymax, ver_L)) # H,W,L
cap = (cap[0] * args.capacity, cap[1] * args.capacity) # scale the capacity

p_generation_time = 0
cost_inference_time = 0
temperature = 1
temp_scale = 1.0
overflow_cost_list = []
edge_via_cost_list = []
pin_via_cost_list = []
start = timeit.default_timer()
for i in range(args.iter):
    if i % int(args.iter/10) == 0 and i != 0:
        temperature = temperature * temp_scale
    p_start = timeit.default_timer()
    hor_p, ver_p = net.forward(temperature)
    p_generation_time += timeit.default_timer() - p_start
    cost_start = timeit.default_timer()
    overflow_cost, edge_via_cost, pin_via_cost, _ = model.layer_assignment_objective(full_end_index,hor_p, ver_p,hor_mask,edge_list, cap, xmax, ymax, args.use_pow, is_steiner, scatter_index, pin_max, pin_min,RoutingRegion3D.hor_first,activation = args.act)
    # print(i,total_cost.cpu().item(), overflow.cpu().item(), via_cost.cpu().item())
    edge_via_cost_list.append(edge_via_cost.cpu().item())
    overflow_cost_list.append(overflow_cost.cpu().item())
    pin_via_cost_list.append(pin_via_cost.cpu().item())
    cost_inference_time += timeit.default_timer() - cost_start
    (edge_via_cost * args.via_coeff + pin_via_cost * args.via_coeff + overflow_cost).backward()
    optimizer.step()
    optimizer.zero_grad()
train_time = timeit.default_timer() - start
# calculate the final cost
discrete_hor_p = hor_p.argmax(dim=1)
# discrete_hor_p to one-hot
discrete_hor_p = torch.zeros(hor_p.shape).to(device).scatter_(1, discrete_hor_p.unsqueeze(1), 1).float()
discrete_ver_p = ver_p.argmax(dim=1)
discrete_ver_p = torch.zeros(ver_p.shape).to(device).scatter_(1, discrete_ver_p.unsqueeze(1), 1).float()
discrete_overflow_cost, discrete_edge_via_cost, discrete_pin_via_cost, max_overflow = model.layer_assignment_objective(full_end_index,discrete_hor_p, discrete_ver_p,hor_mask,edge_list, cap, xmax, ymax, args.use_pow, is_steiner, scatter_index, pin_max, pin_min,RoutingRegion3D.hor_first,activation = 'relu')
print("Final discrete value (overflow, edgevia, pinvia,max_overflow): ", discrete_overflow_cost.cpu().item(), max_overflow.cpu().item(),discrete_edge_via_cost.cpu().item(), discrete_pin_via_cost.cpu().item())


import csv
import os
# create csv file if not exist
if not os.path.exists('./step2.csv'):
    with open('./step2.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # row_list is a list of strings of args
        row_list = []
        for arg in vars(args):
            row_list.append(arg)
        row_list += ['train_time', 'p_generation_time','cost_inference_time','overflow_cost', 'edge_via_cost', 'pin_via_cost','max_overflow', 'peak_gpu_memory (MB)','peak_cpu_memory (MB)']
        writer.writerow(row_list)
# write to csv file 
torch.cuda.empty_cache()
with open('./step2.csv', 'a', newline='') as csvfile:
    writer = csv.writer(csvfile)
    # write each args
    row_list = []
    for arg in vars(args):
        row_list.append(getattr(args, arg))
    row_list += [train_time,p_generation_time, cost_inference_time, discrete_overflow_cost.cpu().item(), discrete_edge_via_cost.cpu().item(), discrete_pin_via_cost.cpu().item(), max_overflow.cpu().item(),torch.cuda.max_memory_allocated()/(1024*1024),tracemalloc.get_traced_memory()[1]/(1024*1024)]
    writer.writerow(row_list)
    

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
plt.figure()
plt.plot(edge_via_cost_list)
plt.plot(pin_via_cost_list)
plt.plot(overflow_cost_list)
plt.legend(["edge via cost",  "pin via cost","overflow cost",])
plt.xlabel("iteration")
plt.ylabel("cost")
plt.savefig("./figs/s2_cost_" + data_name + ".png")
plt.close()

# print runtime
print(p_generation_time, cost_inference_time, edge_via_cost_list[-1], pin_via_cost_list[-1], overflow_cost_list[-1])
# mkdir ./guide if not exist
import os
if not os.path.exists('./guide'):
    os.makedirs('./guide')
file_path = './guide/' + data_name + '.guide'
# argmax to get the layer assignment
# print distribution of hor_p interms of max p value
max_hor_p = hor_p.max(dim=1)[0].cpu().detach().numpy()
max_ver_p = ver_p.max(dim=1)[0].cpu().detach().numpy()

util.visualize_p(max_hor_p,"s2_max_hor_p_" + data_name)
util.visualize_p(max_ver_p,"s2_max_ver_p_" + data_name)

hor_p = hor_p.argmax(dim=1)
ver_p = ver_p.argmax(dim=1)
selected_edge_layer = torch.zeros(edge_list.shape[1], dtype=torch.int64).to(device)
selected_edge_layer[hor_mask] = hor_p
selected_edge_layer[hor_mask == False] = ver_p
selected_edge_layer = selected_edge_layer.cpu().numpy()

# util.write_guide(cap,RoutingRegion3D,file_path, benchmark['net'], edge_list, net_start_idx, net_end_idx, selected_edge_layer, hor_mask, RoutingRegion3D.hor_first,x_min, x_max, y_min, y_max)

