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


import torch
from torch_scatter import scatter_softmax
from torch_scatter import scatter_max
from torch_scatter import scatter_min
import timeit
# define torch NN model
class Net(torch.nn.Module):
    def __init__(self,p_index,pattern_level = 2,device = 'cpu', use_gumble = False, tree_p_index = None, tree_index_per_candidate = None):
        """
        Require:
            p_index: List(int), 
                given the probabilities p, p[pattern_index[i]:pattern_index[i+1]] is the probability distribution for the i_th 2-pin net with multiple candidates (one candidate does not require probability distribution)
                p_index[-1] is the length of the candidate_pool with multiple candidates
            pattern_level: int, the number of candidates for each 2-pin net
            tree_p_index: List(int),
                given the tree probabilities tree_p, tree_p[tree_p_index[i]:tree_p_index[i+1]] is the probability distribution for the trees of i_th net
            tree_index_per_candidate: Tensor(int),
                tree_index_per_candidate[i] is the tree index for the i_th candidate
        """
        super(Net, self).__init__()
        self.p_index = p_index
        repeat = torch.tensor(p_index[1:]) - torch.tensor(p_index[:-1])
        self.p_full_index = torch.repeat_interleave(torch.arange(len(repeat)),repeat).to(device) # self.p_full_index[i] is the 2-pin index for the i_th candidate
        self.pattern_level = pattern_level
        self.tree_p_index = tree_p_index

        # Wei UPDATE 06/08/2023, we noticed that this initialization IS NOT TRULY RANDOM probablity
        # After some calculation, I noticed that pdf of the probablitity after softmax should be f(x) = (1-x)^{n-2} * (n-1), where n is the number of candidates
        # and the cdf is F(x) = 1 - (1-x)^{n-1}
        # therefore, p can be initialized by p = (x)^{(n-1)}, where x is a random number between 0 and 1
        # then, self.p (before softmax) is defined by self.p = log(p) = log(x) * (n-1), here n is 2, so self.p = log(x)
        p_list = []
        for i in range(len(self.p_index)-1):
            p_list.append(torch.log(torch.rand(self.p_index[i+1]-self.p_index[i])) * (self.p_index[i+1]-self.p_index[i]))
        self.p = torch.nn.Parameter(torch.cat(p_list), requires_grad = True) # candidate probability\

        if self.tree_p_index is not None:
            repeat = torch.tensor(self.tree_p_index[1:]) - torch.tensor(self.tree_p_index[:-1])
            self.tree_p_full_index = torch.repeat_interleave(torch.arange(len(repeat)),repeat).to(device) # self.tree_p_full_index[i] is the net index for the i_th tree candidate
            tree_p_list = []
            for i in range(len(self.tree_p_index)-1):
                tree_p_list.append(torch.log(torch.rand(self.tree_p_index[i+1]-self.tree_p_index[i])) * (self.tree_p_index[i+1]-self.tree_p_index[i]))
            self.tree_p = torch.nn.Parameter(torch.cat(tree_p_list), requires_grad = True) # tree_p size is # of trees
            self.tree_index_per_candidate = tree_index_per_candidate
        self.use_gumble = use_gumble


    def forward(self,t, tree_t = None):
        """
        Require:
            t: temperature, float, higher temperature gives more randomness
            tree_t: temperature for tree probability, float, higher temperature gives more randomness
        Return:
            out: Tensor(float), size = len(self.p), the probability distribution for the 2-pin net candidates
            candidate_p: Tensor(float), size = len(self.p), Original probability distribution for the 2-pin net candidates before multiply with tree_p
            tree_p: Tensor(float), size = len(self.tree_p), the probability distribution for the tree candidates
        """
        tree_t = t if tree_t is None else tree_t

        if self.use_gumble:
            gumbels = (-torch.empty_like(self.p, memory_format=torch.legacy_contiguous_format).exponential_().log())
            new_logits = (self.p + gumbels) / t
        else:
            new_logits = self.p / t
        out = scatter_softmax(new_logits - new_logits.max(), self.p_full_index)
        candidate_p = torch.clone(out)
        # dot product with tree_p_repeat if tree_p_index is not None
        tree_p = None
        if self.tree_p_index is not None:
            if self.use_gumble:
                gumbels = (-torch.empty_like(self.tree_p, memory_format=torch.legacy_contiguous_format).exponential_().log())
                new_logits = (self.tree_p + gumbels) / tree_t
            else:
                new_logits = self.tree_p / tree_t
            tree_p = scatter_softmax(new_logits - new_logits.max(), self.tree_p_full_index)
            tree_p_repeat = tree_p[self.tree_index_per_candidate]
            out = out * tree_p_repeat
        return out, candidate_p, tree_p

class LayerAssignmentNet(torch.nn.Module):
    """
    hor_E is the number of hor edges
    hor_L is the number of hor layers 
    C_tensor is to count the size of each consecutive blocks (each block is a segment actually). segment_hor_mask indicates the direction for each segment
        For example, hor_mask = [ True,True, False, False,True,True, False, False], then C = [2,2,2,2], hor_indicator = [True, False, True, False]
        if hor_mask = [True,True, False, False,  False, False], then C = [2,4] because the number of first consecutive elements are 2, the number of second consecutive elements is 4

    When segment_hor_mask and count is not None, we will do layer assignment per segment
    """
    def __init__(self, hor_E, hor_L, ver_E, ver_L,segment_hor_mask= None, hor_count = None, ver_count = None):
        super(LayerAssignmentNet, self).__init__()
        self.segment_hor_mask = segment_hor_mask
        self.hor_count = hor_count
        self.ver_count = ver_count

        if self.segment_hor_mask is None:
            self.hor_p = torch.nn.Parameter(torch.log(torch.rand(hor_E, hor_L) * (hor_L-1)), requires_grad = True)
            self.ver_p = torch.nn.Parameter(torch.log(torch.rand(ver_E, ver_L) * (ver_L-1)), requires_grad = True)
        else:

            self.hor_p = torch.nn.Parameter(torch.log(torch.rand(self.segment_hor_mask.sum(), hor_L) * (hor_L-1)), requires_grad = True)
            self.ver_p = torch.nn.Parameter(torch.log(torch.rand((self.segment_hor_mask == False).sum(), ver_L) * (ver_L-1)), requires_grad = True)
        
    
    """
    Require:
        t: tempreture
    Return:
        hor_out, the probability distribution of layer assignment for each edge
    """
    def forward(self, t):
        hor_out = (self.hor_p/t - (self.hor_p/t).max()).softmax(dim = 1)
        ver_out = (self.ver_p/t - (self.ver_p/t).max()).softmax(dim = 1)
        if self.segment_hor_mask is not None:
            # REPEAT_INTERLEAVE by hor_count and ver_count
            hor_out = torch.repeat_interleave(hor_out, self.hor_count, dim = 0)
            ver_out = torch.repeat_interleave(ver_out, self.ver_count, dim = 0)
        return hor_out, ver_out

    
"""
calculate the objective for layer assignment
Require:
    full_end_index: bool (E) full_end_index[i] = True if i_th edge is the end edge
    hor_out: torch.tensor, (horE, horL), the probability distribution of layer assignment for each horizontal edge,
    is_hor: bool (E): whether the edge is a horizontal 
    edge_path, torch.tensor (2,E), the first is the x_idx, the second row is the y_idx
    weight: (E3), in third case, each connection has a weight. For example, if a steiner points connects three edge, then we set the weight as 1/2. (the actual via cost is 1/2 of sum via cost of all edge pairs)
    via_cost_label: a tuple (turning_via_cost, nonturning_via_cost), the cost of via for turning/nonturning edges.
        turning_via_cost: (L * L): turning_via_cost[i,j] is the via cost when the first hor edge is assigned to layer i and the next ver edge is assigned to layer j
        nonturning_via_cost: (L * L): nonturning_via_cost[i,j] is the via cost when the edge is assigned to layer i and the next edge is assigned to layer j
    cap: a tuple (hor_cap, ver_cap), the capacity of horizontal and vertical layers
        hor_cap: (hor_L, xmax * ymax - 1): hor_cap[i] is the capacity of horizontal layer i
        ver_cap: (ver_L, xmax - 1 * ymax): dir_cap[i] is the capacity of vertical layer i
    use_pow: whether use pow form to calculate the overflow cost
    Otehrs refer to layer_assignment.py
"""

def layer_assignment_objective(full_end_index,hor_out, ver_out,hor_mask, edge_path, cap, xmax, ymax,use_pow, is_steiner, scatter_index, pin_max, pin_min, hor_first,activation = 'relu'):
    hor_p = torch.zeros(hor_out.shape[0], hor_out.shape[1] + ver_out.shape[1], device = hor_out.device) # (hor_E, L), L is the sum of hor_L and ver_L, i.e., total number of layers
    ver_p = torch.zeros(ver_out.shape[0], hor_out.shape[1] + ver_out.shape[1], device = hor_out.device) # (ver_E, L), L is the sum of hor_L and ver_L, i.e., total number of layers
    # map p for horizontal edges and vertical edges to a single probability tensor (full_p)
    full_p = torch.zeros(hor_out.shape[0] + ver_out.shape[0], hor_out.shape[1] + ver_out.shape[1], device =hor_out.device) # (E, L), L is the sum of hor_L and ver_L, i.e., total number of layers
    
    if hor_first == False: # NOTE: hor_first is False, means the second layer is horizontal, which is the real first routable layer. then, then full_p[:,0,2,4,...] is hor_out, full_p[:,1,3,5,...] is ver_out
        hor_indices = torch.arange(0, full_p.shape[1], 2, device = hor_out.device)
        ver_indices = torch.arange(1, full_p.shape[1], 2, device = hor_out.device)
    else:
        hor_indices = torch.arange(1, full_p.shape[1], 2, device = hor_out.device)
        ver_indices = torch.arange(0, full_p.shape[1], 2, device = hor_out.device)
    hor_p[:,hor_indices] = hor_out
    ver_p[:,ver_indices] = ver_out
    full_p[hor_mask == True] = hor_p
    full_p[hor_mask == False] = ver_p

    via_indicator = (full_end_index[:-1] == False)
    # UPDATE 07/18: We notice previous formulation is easily trapped in fake local minimum, so we use mean directly, which is faster to calculate and converge
    mean_edge_layer = torch.einsum('ij,j->i', full_p, torch.arange(0, full_p.shape[1], device = hor_out.device,dtype = torch.float32)) + 1 # (E), +1 here because it starts from the second layer
    # via_cost 1: edge via cost
    edge_pair1 = mean_edge_layer[:-1][via_indicator] # (E1)
    edge_pair2 = mean_edge_layer[1:][via_indicator] # (E1)``
    edge_via_cost = torch.abs(edge_pair1 - edge_pair2).sum()

    
    # via_cost 2: pin via cost
    # full_p : (E,L), mean_edge_layer = (E), and mean_edge_layer[i] = full_p[i][j] * j  
    
    max_edge_layer_per_pin = scatter_max(mean_edge_layer, scatter_index, dim = 0, dim_size = pin_max.shape[0] + 1)[0][:-1] # (P), the last element of scatter_max is the dummy pin
    min_edge_layer_per_pin = scatter_min(mean_edge_layer, scatter_index, dim = 0, dim_size = pin_min.shape[0] + 1)[0][:-1] # (P), the last element is the dummy pin
    pin_via_cost = (torch.max(torch.stack((max_edge_layer_per_pin,pin_max)),dim = 0).values - torch.min(torch.stack((min_edge_layer_per_pin,pin_min)),dim = 0).values).sum() # (P), the last element is the dummy pin

    # now calculate the overflow cost 
    hor_edges = edge_path[:,hor_mask] # 2, HorE
    hor_demand = torch.sparse_coo_tensor(hor_edges,hor_out,(xmax, ymax-1, hor_out.shape[1])).coalesce().to_dense() # (xmax, ymax-1, hor_L)
    # swap dim 0,1,2 to 2,0,1
    ver_edges = edge_path[:,hor_mask == False]
    ver_demand = torch.sparse_coo_tensor(ver_edges,ver_out,(xmax-1, ymax, ver_out.shape[1])).coalesce().to_dense() # (xmax-1, ymax, ver_L)
    if use_pow:
        overflow_cost = (1 / (1 + torch.exp(cap[0]-hor_demand))).sum() + (1 / (1 + torch.exp(cap[1]-ver_demand))).sum() 
    else:
        if activation == 'celu':
            act = torch.nn.CELU()
        elif activation == 'relu':
            act = torch.nn.ReLU()
        elif activation == 'leaky_relu':
            act = torch.nn.LeakyReLU()
        elif activation == 'exp':
            def exp(x):
                return torch.exp(x/5)
            act = exp
        else:
            raise NotImplementedError
        overflow_cost = act(hor_demand -cap[0]).sum() # (hor_L, xmax * ymax - 1)
        overflow_cost += act(ver_demand - cap[1]).sum() # (ver_L, xmax - 1 * ymax)
    max_overflow = max(torch.nn.ReLU()(hor_demand -cap[0]).max(),torch.nn.ReLU()(ver_demand - cap[1]).max())
    return overflow_cost, edge_via_cost, pin_via_cost,max_overflow




# define objective function
def objective_function(RoutingRegion, hor_path, ver_path,wire_length_count, via_info, p, args,activation = None,iteration=None, hor_batch_size=None,ver_batch_size=None, hor_shuffled_indices=None,ver_shuffled_indices = None):
    """
    the objective function is calculated by:
    1. overflow cost, i.e., sum relu (sum (p[i] * (hor_path[i])) - RoutingRegion.cap_mat[0])) + sum relu sum (p[i] * (candidate_pool[p_index2pattern_index[i]][0][1]) - RoutingRegion.cap_mat[1])
    Require:
        RoutingRegion: routing_region object
        hor_path: sparse tensor, (xmax * ymax - 1, n), n is the number of candidates
        ver_path: sparse tensor, (xmax - 1 * ymax, n), n is the number of candidates
        via_info: (via_map, via_count) 
            via_map: sparse tensor, (xmax * ymax, n), n is the number of candidates
            via_count: tensor, (n), n is the number of candidates
        wire_length_count: tensor, (n), n is the number of candidates
        p: tensor, (n), p[i] is the probability of the i_th candidate to be selected
        pow: bool, whether to use a power based function to calculate overflow rather than relu
        add_via_in_overflow: bool, whether to add via congestion in overflow cost (demand = wire cost + \sqrt((via(u) + via (v))/2))
    """
    pow = args.use_pow
    add_via_in_overflow = args.add_via
    if activation is None:
        activation = args.act
    # calculate overflow cost by tensor operation directly
    hor_cap = RoutingRegion.cap_mat[0].flatten()
    ver_cap = RoutingRegion.cap_mat[1].flatten()

    via_map,via_count = via_info
    hor_demand = torch.matmul(hor_path,p) # the demand for the horizontal edge, if add_via_in_overflow is False, we only count the wire congestion
    ver_demand = torch.matmul(ver_path,p) # the demand for the vertical edge
    if add_via_in_overflow:
        xmax = RoutingRegion.xmax
        ymax = RoutingRegion.ymax
        this_via_map = torch.matmul(via_map,p).view(xmax,ymax) # (xmax, ymax)
        # hor_via = torch.sqrt((via_map[:,:-1] + via_map[:,1:])/2).flatten()
        # ver_via = torch.sqrt((via_map[:-1,:] + via_map[1:,:])/2).flatten()
        hor_via = ((this_via_map[:,:-1] + this_via_map[:,1:]) * args.via_layer / 2).flatten() # /2 because layers are half divided by hor/ver
        ver_via = ((this_via_map[:-1,:] + this_via_map[1:,:]) * args.via_layer / 2).flatten()
        hor_demand = hor_via + hor_demand
        ver_demand = ver_via + ver_demand

    hor_total_elements = hor_cap.numel()
    ver_total_elements = ver_cap.numel()
    if iteration is not None and hor_batch_size is not None and hor_shuffled_indices is not None:
        hor_start_idx = iteration * hor_batch_size
        hor_end_idx = hor_start_idx + hor_batch_size
        hor_end_idx = min(hor_end_idx, hor_total_elements)
        hor_selected_indices = hor_shuffled_indices[hor_start_idx:hor_end_idx]
        hor_mask = torch.zeros(hor_total_elements, dtype=torch.bool)
        hor_mask[hor_selected_indices] = 1
        # vertical
        ver_start_idx = iteration * ver_batch_size
        ver_end_idx = ver_start_idx + ver_batch_size
        ver_end_idx = min(ver_end_idx, ver_total_elements)
        ver_selected_indices = ver_shuffled_indices[ver_start_idx:ver_end_idx]
        ver_mask = torch.zeros(ver_total_elements, dtype=torch.bool)
        ver_mask[ver_selected_indices] = 1
    else:
        hor_mask = torch.ones(hor_total_elements, dtype=torch.bool)
        ver_mask = torch.ones(ver_total_elements, dtype=torch.bool)
    
    hor_demand = hor_demand[hor_mask]
    ver_demand = ver_demand[ver_mask]
    hor_cap = hor_cap[hor_mask]
    ver_cap = ver_cap[ver_mask]

    if pow:
        overflow_cost = torch.sum(1 / (1 + torch.exp(hor_cap-hor_demand))) + torch.sum(1 / (1 + torch.exp(ver_cap-ver_demand)))
    else:
        if activation == 'celu':
            act = torch.nn.CELU(alpha = 2.0)
        elif activation == 'relu':
            act = torch.nn.ReLU()
        elif activation == 'leaky_relu':
            act = torch.nn.LeakyReLU()
        elif activation == 'exp':
            def exp(x):
                return torch.exp(x/5)
            act = exp
        else:
            raise NotImplementedError
        overflow_cost = torch.sum(act(- hor_cap + hor_demand )) + torch.sum(act(- ver_cap + ver_demand))
        max_overflow = max(torch.nn.ReLU()(- hor_cap + hor_demand).max(),torch.nn.ReLU()(- ver_cap + ver_demand).max())
    via_cost = (via_count * p).sum()
    wire_length_cost = (wire_length_count * p).sum()
    return overflow_cost, via_cost, wire_length_cost, max_overflow, hor_demand - hor_cap, ver_demand - ver_cap

def discrete_objective_function(RoutingRegion, hor_path, ver_path,wire_length_count, via_info, selected_index, args):
    """
    the discrete objective function.
    The difference with objective_function is that the input is a list of selected index rather than p
    Require:
        selected_index: Tensor(int), selected_index[i] is the index of the selected candidate for the i_th 2-pin net
        max(selected_index) < # of candidates
        (Others refer to objective_function)
    """
    hor_cap = RoutingRegion.cap_mat[0].flatten()
    ver_cap = RoutingRegion.cap_mat[1].flatten()
    via_map,via_count = via_info
    selected_hor_path = torch.index_select(hor_path,1,selected_index)
    selected_ver_path = torch.index_select(ver_path,1,selected_index)
    hor_demand = torch.sum(selected_hor_path,dim = 1) # the demand for the horizontal edge, if add_via_in_overflow is False, we only count the wire congestion
    ver_demand = torch.sum(selected_ver_path,dim = 1) # the demand for the vertical edge
    pow = args.use_pow
    add_via_in_overflow = args.add_via
    
    if add_via_in_overflow:
        xmax = RoutingRegion.xmax
        ymax = RoutingRegion.ymax
        via_congestion = torch.index_select(via_map,1,selected_index).sum(dim = 1).to_dense().view(xmax,ymax) # shape (xmax * ymax)
        hor_via = ((via_congestion[:,:-1] + via_congestion[:,1:]) * args.via_layer / 2).flatten() # shape (xmax * ymax - 1)
        ver_via = ((via_congestion[:-1,:] + via_congestion[1:,:]) * args.via_layer / 2).flatten() # shape (xmax - 1 * ymax)
        hor_demand = hor_via + hor_demand
        ver_demand = ver_via + ver_demand
    if pow:
        overflow_cost = torch.sum(1 / (1 + torch.exp(hor_cap-torch.sum(selected_hor_path,dim = 1)))) + torch.sum(1 / (1 + torch.exp(ver_cap-torch.sum(selected_ver_path,dim = 1))))
    else:
        overflow_cost = torch.sum(torch.relu(- hor_cap + torch.sum(selected_hor_path,dim = 1) )) + torch.sum(torch.relu(- ver_cap + torch.sum(selected_ver_path,dim = 1)))
    
    wl_cost = wire_length_count[selected_index].sum()
    via_cost = via_count[selected_index].sum()
    max_overflow = max(torch.nn.ReLU()(- hor_cap + torch.sum(selected_hor_path,dim = 1)).max(),torch.nn.ReLU()(- ver_cap + torch.sum(selected_ver_path,dim = 1)).max())
    return overflow_cost, wl_cost, via_cost, max_overflow

"""
BACKUP

    # via_cost 1: edge via cost
    edge_pair1 = full_p[:-1][via_indicator] # (E1, L)
    edge_pair2 = full_p[1:][via_indicator] # (E1, L)``
    # outer product of p[:-1] and p[1:]
    outer_product = torch.einsum('ij,ik->ijk', edge_pair1, edge_pair2) # (E1, L, L)
    # cost_label is a tensor with shape L, L, and cost_label[i][j] = abs(i-j)
    cost_label = abs(torch.arange(0, full_p.shape[1], device = full_p.device,dtype = torch.float32).view(-1,1) - torch.arange(0, full_p.shape[1], device = full_p.device).view(1,-1))
    # calculate the via cost
    edge_via_cost = torch.einsum('ijk,jk', outer_product, cost_label).sum() # non-turning cost
    

    start = timeit.default_timer()
    p = torch.zeros(hor_out.shape[0] + ver_out.shape[0], max(hor_out.shape[1], ver_out.shape[1])).to(hor_out.device) # (E, L), L is the max between hor_L and ver_L
    p[hor_mask == True, :hor_out.shape[1]] = hor_out # p: torch.tensor, (E, L), the probability distribution of layer assignment for each horizontal edge, E is the number of gcell edges. L is the number of layers/2.
    p[hor_mask == False, :ver_out.shape[1]] = ver_out
    step1 = timeit.default_timer() - start
    # via_cost 1:
    edge_pair1 = p[:-1][via_indicator1] # (E1, L)
    edge_pair2 = p[1:][via_indicator1] # (E1, L)``
    # outer product of p[:-1] and p[1:]
    outer_product = torch.einsum('ij,ik->ijk', edge_pair1, edge_pair2) # (E1, L, L)
    # calculate the via cost
    via_cost = torch.einsum('ijk,jk', outer_product, via_cost_label[0]).sum() # non-turning cost

    # via_cost 2:
    edge_pair1 = p[via_indicator2[0]] # (E2, L)
    edge_pair2 = p[via_indicator2[1]] # (E2, L)
    # outer product of p[:-1] and p[1:]
    outer_product = torch.einsum('ij,ik->ijk', edge_pair1, edge_pair2) # (E2, L, L)
    # calculate the via cost
    via_cost += torch.einsum('ijk,jk', outer_product, via_cost_label[1]).sum() # turning cost
    end = timeit.default_timer()
    print('via_cost time: ', end - start, '; step 1 time: ', step1)


    same_dir_pairs =  hor_mask[via_indicator3[0]] == hor_mask[via_indicator3[1]]
    same_dir_edge_pair1 = p[via_indicator3[0]][same_dir_pairs] # (E3-same-dir, L)
    same_dir_edge_pair2 = p[via_indicator3[1]][same_dir_pairs] # (E3-same-dir, L)
    # outer product of p[:-1] and p[1:]
    outer_product = torch.einsum('ij,ik->ijk', same_dir_edge_pair1, same_dir_edge_pair2) * weight[same_dir_pairs].view(-1,1,1).float() # (E3-same-dir, L, L)
    # calculate the via cost
    via_cost += torch.einsum('ijk,jk', outer_product, via_cost_label[0]).sum() # non-turning cost

    # hor -> ver
    hor2ver_pairs = (hor_mask[via_indicator3[0]] == True) * (hor_mask[via_indicator3[1]] == False)
    hor2ver_edge_pair1 = p[via_indicator3[0]][hor2ver_pairs] # (E3-hor2ver, L)
    hor2ver_edge_pair2 = p[via_indicator3[1]][hor2ver_pairs] # (E3-hor2ver, L)
    hor2ver_weight = weight[hor2ver_pairs].view(-1,1,1).float()

    # ver -> hor
    ver2hor_pairs = (hor_mask[via_indicator3[0]] == False) * (hor_mask[via_indicator3[1]] == True)
    ver2hor_edge_pair1 = p[via_indicator3[1]][ver2hor_pairs] # (E3-ver2hor, L)
    ver2hor_edge_pair2 = p[via_indicator3[0]][ver2hor_pairs] # (E3-ver2hor, L)
    ver2hor_weight = weight[ver2hor_pairs].view(-1,1,1).float()

    turning_weight = torch.cat([hor2ver_weight, ver2hor_weight], dim=0) # (E3-hor2ver + E3-ver2hor, 1, 1)
    edge_pair1 = torch.cat([hor2ver_edge_pair1, ver2hor_edge_pair1], dim=0) # (E3-hor2ver + E3-ver2hor, L)
    edge_pair2 = torch.cat([hor2ver_edge_pair2, ver2hor_edge_pair2], dim=0) # (E3-hor2ver + E3-ver2hor, L)
    # outer product of p[:-1] and p[1:]
    outer_product = torch.einsum('ij,ik->ijk', edge_pair1, edge_pair2) * turning_weight # (E3-same-dir, L, L)
    # calculate the via cost
    via_cost += torch.einsum('ijk,jk', outer_product, via_cost_label[0]).sum() # non-turning cost

"""
