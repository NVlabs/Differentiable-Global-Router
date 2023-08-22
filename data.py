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
The script includes
1. I/O functions
2. basic data class
    2.1 Net, which includes pins
    2.2 Pin, (x,y,layer,parent_pin)
"""
import timeit
import random
import torch
import numpy as np
# Net class
class Net:
    """
    net_name: string, net name
    net_ID: int, net ID
    pins: list of Pin objects. pins[i][j] is the j_th Pin (PinID = j) for i_th tree
    need_route: bool, whether this net needs to be routed, if not, then we set list of gcell (x,y, layeridx)
    gcell_list: list of gcell (x,y, layeridx), if need_route is False, then we set gcell_list
    num_pins: number of physical pins, (WITHOUT steiner points), physical pins is not Pin object, (Pin object is gcell actually)
    """
    def __init__(self, net_name, net_ID, need_route= True, pins = None,num_pins = None, gcell_list = None):
        self.net_name = net_name
        self.net_ID = net_ID
        self.pins = [] if pins is None else pins
        self.num_pins = num_pins
        self.need_route = need_route
        self.gcell_list = gcell_list
        if self.need_route == False:
            assert(self.gcell_list is not None)


    # tree index to add the pin
    def add_pin(self, pin ,index = -1):
        # make sure new pin is not in the list in terms of the location
        for p in self.pins[index]:
            if p.x == pin.x and p.y == pin.y:
                return
        self.pins[index].append(pin)


class Pin:
    """
    Gcell object
    x: int, x coordinate
    y: int, y coordinate
    layer: int, layer index
    parent_pin: int, parent pin index
    physical_pin_layers: list of int, layers of physical pins. len(physical_pin_layers) = 2, lower layer and upper layer
    """
    def __init__(self, x, y, layer = None, parent_pin = None, is_steiner = False,child_pins = None, physical_pin_layers = None):
        self.x = x
        self.y = y
        self.layer = layer
        self.parent_pin = parent_pin
        self.is_steiner = is_steiner
        self.child_pins = child_pins
        self.physical_pin_layers = physical_pin_layers

    def set_parent(self, parent_pin):
        self.parent_pin = parent_pin
    
    def add_child(self, child_pin):
        if self.child_pins is None:
            self.child_pins = []
        self.child_pins.append(child_pin)


class routing_region:
    """
    The routing region storing current status
    xmax: int, x max
    ymax: int, y max
    cap_mat: (hor, ver) OR float, which will assign a flat float value to each gcell. Otherwise, the default cap will be 1
        hor: tensor, xmax * (ymax - 1)
        ver: tensor, (xmax - 1) * ymax

    """
    def __init__(self, xmax, ymax, cap_mat = None,device = 'cpu'):
        self.xmax = xmax
        self.ymax = ymax
        if cap_mat is None:
            self.cap_mat = (torch.ones((xmax, ymax-1)).to(device), torch.ones((xmax - 1, ymax)).to(device)) 
        # elif cap_mat is a integer, assign a flat value to each gcell
        elif isinstance(cap_mat, float) or isinstance(cap_mat, int):
            self.cap_mat = (torch.ones((xmax, ymax-1)).to(device) * cap_mat, torch.ones((xmax - 1, ymax)).to(device) * float(cap_mat))
        else:
            self.cap_mat = (cap_mat[0].to(device), cap_mat[1].to(device))
        
    def to(self,device):
        self.cap_mat = [self.cap_mat[0].to(device), self.cap_mat[1].to(device)]


# class 3D_routing_region, child class of routing_region
class routing_region_3D(routing_region):
    """
    Besides the attributes of routing_region, it also has cap_mat_3D and hor_fist
    cap_mat_3D is a tuple of (hor_cap_list, ver_cap_list)
        hor_cap_list: list of tensor (xmax * (ymax - 1)), each tensor is a cap_mat of a layer 
        ver_cap_list: list of tensor ((xmax - 1) * ymax), each tensor is a cap_mat of a layer
    hor_fist is a boolean value, which indicates whether the horizontal layer is the first layer
    """
    def __init__(self, xmax, ymax, cap_mat_3D, hor_first):
        self.cap_mat_3D = cap_mat_3D
        self.hor_first = hor_first
        super().__init__(xmax, ymax, cap_mat = None)
    def to(self,device):
        self.cap_mat_3D = ([self.cap_mat_3D[0][i].to(device) for i in range(len(self.cap_mat_3D[0]))], [self.cap_mat_3D[1][i].to(device) for i in range(len(self.cap_mat_3D[1]))])
"""
randomly generate net data (with topology, i.e., parent pin).
In the alg, each pin is randomly assigned into a range specified by net_size. The parent pin is also randomly assigned except itself, (therefore, the tree might not be connected)
xmax: int, x max of the routing region
ymax: int, y max of the routing region
num_nets: int, number of nets
max_num_pin: int, number of pins per net (range: 2 ~ num_pin)
net_size: (int,int), net size (x,y) value. which is the max 

return: list of Net objects
"""
from typing import Tuple
def random_data(xmax: int, ymax: int, num_nets: int, max_num_pin: int, net_size: Tuple[int, int],seed = None):
    if seed is not None:
        random.seed(seed)
    # generate net data
    net_data = []
    for net_ID in range(num_nets):
        net_name = 'net_' + str(net_ID)
        net = Net(net_name, net_ID)
        # randomly generate pins
        num_pin = random.randint(2, max_num_pin)
        for pin_ID in range(num_pin):
            # randomly generate location for bottom left corner
            x = random.randint(0, xmax - net_size[0])
            y = random.randint(0, ymax - net_size[1])
            # randomly generate pin size
            x_size = random.randint(1, net_size[0])
            y_size = random.randint(1, net_size[1])
            # randomly generate parent pin
            parent_pin = None if pin_ID == 0 else net.pins[-1]
            # add pin
            net.add_pin(Pin(min(x + x_size,xmax-1), min(y + y_size,ymax-1), parent_pin = [parent_pin]))
        net.update_num_tree(1)
        net_data.append(net)
    return net_data

"""
Given the candidate pool by get_initial_candidate_pool(),
get the sparse path tensor
Require:
    candidate_pool: list of candidate pool for each net
Return:
    p_index: List(int), 
        given the candidate probabilities p, p[p_index[i]:p_index[i+1]] is the probability distribution for the i_th 2-pin net with multiple candidates (one candidate does not require probability distribution)
        p_index[-1] is the length of the candidate_pool with multiple candidates
    p_index_full: Tensor(n), n is the number of candidates
        p_index_full[i] is j if p_index[j] <= i < p_index[j+1]
    p_index2pattern_index: List((int,int,int,int)),
        the map from index in probability array p to the index in candidate_pool

    tree_p_index: List(int),
        given the tree probabilities tree_p, tree_p[tree_p_index[i]:tree_p_index[i+1]] is the probability distribution for the trees of i_th net
    tree_index_per_candidate: Tensor(int),
        tree_index_per_candidate[i] is tree index for i_th candidate
    tree_p_index2pattern_index: List((int,int)), the map from index in tree probability array tree_p to the index in candidate_pool
    
    hor_path: sparse tensor, (xmax * (ymax - 1), n), n is the number of candidates
    ver_path: sparse tensor, ((xmax - 1) * ymax, n), n is the number of candidates
    via_info: (via_map, via_count) 
        via_map: sparse tensor, (xmax * ymax, n), n is the number of candidates
        via_count: tensor, (n), n is the number of candidates
    wire_length_count: tensor, (n), n is the number of candidates
"""
def process_pool(candidate_pool,xmax, ymax,device = 'cpu'):
    start = timeit.default_timer()
    # create hor_path (sparse_tensor, (n, xmax, ymax - 1), n is the number of candidates)
    via_map_list = []
    wl_list = []
    hor_list = []
    ver_list = []
    p_index = [0]
    tree_p_index = [0] 
    tree_index_per_candidate = []
    p_index2pattern_index = []
    tree_p_index2pattern_index = []
    for idx1, net in enumerate(candidate_pool):
        tree_p_index.append(tree_p_index[-1] + len(net))
        for idx2, tree in enumerate(net):
            tree_p_index2pattern_index.append((idx1,idx2))
            for idx3,two_pin_net in enumerate(tree):
                if len(two_pin_net) == 0:
                    continue
                p_index.append(p_index[-1] + len(two_pin_net[1]))
                for idx4, candidate in enumerate(two_pin_net[1]): # two_pin_net[0] is the tuple of (pin, parent_pin)
                    p_index2pattern_index.append((idx1,idx2,idx3,idx4))
                    wl_list.append(candidate[0])
                    # extract via_map
                    via_map = candidate[1] # np array (2, num_via)
                    via_map = via_map[0] * ymax + via_map[1] # np array (num_via)
                    via_map = torch.sparse_coo_tensor(via_map.reshape(1,-1), torch.ones(via_map.shape[0], dtype=torch.float32,device=device), (xmax * ymax,),device=device) # sparse tensor (xmax * ymax)
                    via_map_list.append(via_map)

                    #extract hor_path
                    edge_index, is_hor = candidate[2] # edge_index is np array (2, num_edge), is_hor is np array (num_seg)
                    hor_edge_index = edge_index[:,is_hor] # np array (2, num_hor_edge)
                    hor_edge_index = hor_edge_index[0] * (ymax - 1) + hor_edge_index[1] # np array (num_hor_edge)
                    hor_edge_index = torch.sparse_coo_tensor(hor_edge_index.reshape(1,-1), torch.ones(hor_edge_index.reshape(1,-1).shape[1], dtype=torch.float32), (xmax * (ymax - 1),)) # sparse tensor (xmax * (ymax - 1))
                    hor_list.append(hor_edge_index)

                    # extract ver_path
                    ver_edge_index = edge_index[:,~is_hor] # np array (2, num_ver_edge)
                    ver_edge_index = ver_edge_index[0] * ymax + ver_edge_index[1] # np array (num_ver_edge)
                    assert((ver_edge_index < (xmax - 1) * ymax).all())
                    ver_edge_index = torch.sparse_coo_tensor(ver_edge_index.reshape(1,-1), torch.ones(ver_edge_index.reshape(1,-1).shape[1], dtype=torch.float32), ((xmax - 1) * ymax,)) # sparse tensor ((xmax - 1) * ymax)
                    ver_list.append(ver_edge_index)
                    tree_index_per_candidate.append(len(tree_p_index2pattern_index) - 1)

    p_index_full = torch.zeros(p_index[-1], dtype=torch.long,device=device)   
    for i in range(len(p_index) - 1):
        p_index_full[p_index[i]:p_index[i+1]] = i
    
    tree_p_index_full = torch.zeros(tree_p_index[-1], dtype=torch.long,device=device)
    for i in range(len(tree_p_index) - 1):
        tree_p_index_full[tree_p_index[i]:tree_p_index[i+1]] = i

    wire_length_count = torch.tensor(wl_list).to(device)
    via_map = torch.stack(via_map_list,dim = 1).to(device).float()
    via_count = via_map.sum(dim = 0).to_dense() # (n)
    hor_path = torch.stack(hor_list,dim = 1).to(device)
    ver_path = torch.stack(ver_list,dim = 1).to(device)
    print("Sparse tensor generation time: ", timeit.default_timer() - start)
    tree_index_per_candidate = torch.tensor(tree_index_per_candidate).to(device)
    return p_index, p_index_full, np.array(p_index2pattern_index), hor_path, ver_path, wire_length_count, (via_map,via_count), tree_p_index, tree_index_per_candidate, np.array(tree_p_index2pattern_index), tree_p_index_full



import matplotlib.pyplot as plt
import seaborn as sns
import os
# plot p distribution
"""
Require:
    p: numpy(float), the probability distribution for each candidate
"""
def plot_p_dist(p,path = './figs/p_distribution.png'):
    # plot distribution of p
    # path folder does not exist, create it
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    sns.set()
    plt.figure()
    plt.hist(p.detach().cpu().numpy(), bins = 10)
    plt.savefig(path)
    plt.close()


