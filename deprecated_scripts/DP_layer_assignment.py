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
The Dynamic Programming version for the layer assignment problem.
"""

import numpy as np
import argparse
import torch
import timeit
import util

"""
mvc: final result. mvc[v][r] = minimum via cost of the subtree rooted at vertex v when parent edge of v is assigned to layer r.
    mvc[v][r] = inf means it is not yet calculated
    # mvc[v][r] = min_{r_u \in layers}{ \sum_{u \in child of v} mvc[u][r_u] + vc(u)}, vc(u) is the via cost = max_(r_u, r_v) - min_(r_u, r_v)
net: the netlist

capacity: the remaining capacity of the routing region, which is used to determine whether layer is routable
net_start_edge_idx: the start edge idx for this net, used for shift
target_edge_idx: the index of the target edge in the net (each edge is a "node" in the DP paper)
layer_range: iterable, the range of layers to be considered
start_idx_set: a map to search whether the edge idx is the start idx of a 2-pin net
end_idx_set: a map to search whether the edge idx is the end idx of a 2-pin net
"""
def cal_mvc(mvc,net,capacity,edge_list,net_start_edge_idx,target_edge_idx,hor_mask_set,hor_layer_range,ver_layer_range,start_idx_set,end_idx_set):
    # not in end, means it is a single path.
    if target_edge_idx not in end_idx_set:
        layer_range = hor_layer_range if target_edge_idx in hor_mask_set else ver_layer_range
        for target_layer in layer_range:
            if capacity[edge_list[0,target_edge_idx],edge_list[1,target_edge_idx],target_layer] <= 0:
                # not routable, directly calculate mvc as 1e9
                mvc[target_edge_idx - net_start_edge_idx][target_layer] = 1e9
            else:
                current_min = np.inf
                child_layer_range = hor_layer_range if target_edge_idx + 1 in hor_mask_set else ver_layer_range
                for child_layer in child_layer_range:
                    # target_edge_idx + 1 - net_start_edge_idx is the idx of the child edge in the mvc matrix, == inf means it is not calculated
                    if mvc[target_edge_idx + 1 - net_start_edge_idx][child_layer] == np.inf:
                        cal_mvc(mvc,net,capacity,edge_list,net_start_edge_idx,target_edge_idx + 1,hor_mask_set,hor_layer_range,ver_layer_range,start_idx_set,end_idx_set)
                    assert(mvc[target_edge_idx + 1 - net_start_edge_idx][child_layer] != np.inf)
                    current_min = min(current_min, mvc[target_edge_idx + 1 - net_start_edge_idx][child_layer] + np.abs(child_layer - target_layer)) # child_layer - target_layer is the via cost
                mvc[target_edge_idx - net_start_edge_idx][target_layer] = current_min


# I decied to draw a blueprint and design a directed graph datastructure to represent the gcell graphs.


parser = argparse.ArgumentParser()
# args.data_path, str, the path for contest benchmark, when value is None, then, use random data
parser.add_argument('--data_path', type=str, default='$cugr2/run/ispd18_test1.pt')
args = parser.parse_args()

data_name = args.data_path.split("/")[-1].split(".")[0]

# pin2edge_idx is a map from (netname, child_pinidx) to corresponding start and end edge index
pin2edge_idx, net_start_idx, net_end_idx, edge_list, is_hor_list, end_index, start_idx, edge_idx1, edge_idx2, weights, is_steiner, scatter_index, pin_max, pin_min= torch.load("./2D_result/%s.pt"%data_name)

benchmark = torch.load(args.data_path)
hor_mask = np.array(is_hor_list)

RoutingRegion3D = benchmark["region3D"]
xmax = RoutingRegion3D.xmax
ymax = RoutingRegion3D.ymax
hor_E = hor_mask.sum()
ver_E = (hor_mask == False).sum()
hor_L = len(RoutingRegion3D.cap_mat_3D[0])
ver_L = len(RoutingRegion3D.cap_mat_3D[1])
cap = (np.stack(RoutingRegion3D.cap_mat_3D[0],dim=1).reshape(xmax, ymax-1,hor_L), np.stack(RoutingRegion3D.cap_mat_3D[1],dim=2).reshape(xmax-1,ymax, ver_L)) # H,W,L

# step1 calculate npv: netlenth / pin_num for each net, and then sort the nets by npv
for net in benchmark['net']:
    net_length =  net_end_idx[net.net_name] - net_start_idx[net.net_name] + 1
    num_pin = net.num_pins
    net.npv = net_length / num_pin
benchmark['net'] = sorted(benchmark['net'], key=lambda x: x.npv, reverse=True)

# step2 layer assignment by dynamic programming
for net in benchmark['net']:
    # first, find the node without parent pin
    find_root = False
    # pins[0] means the first steiner tree, which is the default structure
    for pin_idx, pin in enumerate(net.pins[0]):
        if pin.parent_pin is None:
            root_pin_idx = pin_idx
            find_root = True
            break
    assert find_root, "cannot find root pin"
    # second, do dynamic programming, start from the root pin

    net_length =  net_end_idx[net.net_name] - net_start_idx[net.net_name] + 1
    mvc = np.full((net_length, hor_L + ver_L - 1), np.inf)


