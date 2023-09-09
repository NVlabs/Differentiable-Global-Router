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
Script for utility functions
"""
import os
from typing import List
from data import Net, Pin
# sparse tensor
import torch
from typing import Tuple
import numpy as np
import math
import seaborn as sns
import matplotlib.pyplot as plt
from torch_scatter import scatter_max
import timeit
import random
"""
Given x, y start and end coordinates, return the edge idx, and the number of edges
NOTE: at least one of the start and end coordinates should be the same
Require:
    x_start: int, x start coordinate
    x_end: int, x end coordinate
    y_start: int, y start coordinate
    y_end: int, y end coordinate
Return:
    edge_idx: numpy array (2, edge_num), edge_idx[0,i] is the x-coordinate of the i_th edge
    is_hor: bool numpy array (edge_num), whether the edge is horizontal
    edge_num: int, number of edges
"""
def get_edge_info(x_start, x_end, y_start, y_end):
    assert x_start == x_end or y_start == y_end
    if x_start == x_end:
        edge_num = abs(y_end - y_start)
        edge_idx = np.zeros((2, edge_num), dtype = int)
        edge_idx[0,:] = x_start
        indicator = 1 if y_end > y_start else -1
        # use indicator, edge_idx will be ordered from the specified start to end
        edge_idx[1,:] = np.arange(y_start, y_end, indicator) - int(indicator == -1)
        is_hor = np.ones(edge_num, dtype = bool)
    else:
        edge_num = abs(x_end - x_start)
        edge_idx = np.zeros((2, edge_num), dtype = int)
        indicator = 1 if x_end > x_start else -1
        edge_idx[0,:] = np.arange(x_start, x_end, indicator) - int(indicator == -1)
        edge_idx[1,:] = y_start
        is_hor = np.zeros(edge_num, dtype = bool)
    return edge_idx,is_hor, edge_num


"""
given two pins and the turing point coordinate, return the z pattern routing result
NOTE: this can also generate C-pattern routing result!
The path of z/c pattern is like follows:  pin -> turning point -> turning point on the other side -> parent pin

Require:
    pin: Pin object
    parent_pin: parent Pin object
    turning: Tuple(int,int), turning point coordinate, NOTE either turning[0] == small_x or turning[1] == small_y
    xmax: int, x max of the routing region
    ymax: int, y max of the routing region
Return:
    a Tuple including follows:
    2. path_length (wirelength)
    3. turning_point (via count)
    4. edge_info tuple, edge is the gcell edge
        encoding_matrix: index tensor (2, edge_num), note that the edge should be ordered, 
            THE FIRST EDGE ALWAYS CONNECTS TO THE CHILD PIN
            THE LAST EDGE ALWAYS CONNECTS TO THE PARENT PIN
        is_hor, tensor (edge_num), bool, whether the edge is horizontal
    5. seg_info repeats: tensor, (segment), each element is the number of the edge of the segment
"""
def z_pattern_routing(pin: Pin, parent_pin: Pin, turning: Tuple[int,int], xmax: int, ymax:int):
    # we start from the top pin, (the pin with smaller x coordinate, i.e., top rows)
    small_x = pin.x if pin.x < parent_pin.x else parent_pin.x
    large_x = pin.x if pin.x > parent_pin.x else parent_pin.x
    small_y = pin.y if pin.y < parent_pin.y else parent_pin.y
    large_y = pin.y if pin.y > parent_pin.y else parent_pin.y
    # either turning[0] == small_x or turning[1] == small_y
    assert(turning[0] == small_x or turning[1] == small_y)
    is_x_turing = True if turning[0] == small_x else False # is_x_turing means the y corrdinate of turning point is different with both pins
    if is_x_turing:
        path_length = abs(pin.y - turning[1]) + abs(parent_pin.y - turning[1]) + abs(pin.x - parent_pin.x)
        # step 1, fixed pin x, y from pin y to turning y
        edges1,is_hor1, edge_num1 = get_edge_info(pin.x, pin.x, pin.y, turning[1])
        # step 2, fixed pin y, x from pin x to parent pin x
        edges2,is_hor2, edge_num2 = get_edge_info(pin.x, parent_pin.x, turning[1], turning[1])
        # step 3, fixed parent x, y from parent pin y to turning y
        edges3,is_hor3, edge_num3 = get_edge_info(parent_pin.x, parent_pin.x, turning[1], parent_pin.y)
    else:
        path_length = abs(pin.x - turning[0]) + abs(parent_pin.x - turning[0]) + abs(pin.y - parent_pin.y)
        # step 1, fixed pin y, x from pin x to turning x
        edges1,is_hor1, edge_num1 = get_edge_info(pin.x, turning[0], pin.y, pin.y)
        # step 2, fixed pin x, y from pin y to parent pin y
        edges2,is_hor2, edge_num2 = get_edge_info(turning[0], turning[0], pin.y, parent_pin.y)
        # step 3, fixed parent y, x from parent pin x to turning x
        edges3,is_hor3, edge_num3 = get_edge_info(turning[0], parent_pin.x, parent_pin.y, parent_pin.y)
    edge_idx = np.concatenate((edges1, edges2, edges3), axis = 1)
    is_hor = np.concatenate((is_hor1, is_hor2, is_hor3))
    seg_idx = np.array([edge_num1, edge_num2, edge_num3])
    turning_point = np.array([[pin.x,parent_pin.x],[turning[1],turning[1]]]) if is_x_turing else np.array([[turning[0],turning[0]],[pin.y,parent_pin.y]])
    return (path_length, turning_point, (edge_idx, is_hor), seg_idx)



"""
now, (06/04/2023), we only consider two l-shape routing
UPDATE (06/09/2023), we add z-shape routing
Require:
    pin: Pin object
    parent_pin: parent Pin object
    xmax: int, x max of the routing region
    ymax: int, y max of the routing region
    (opt) pattern_level: 1 only l-shape, 2 add z shape 3 add c shape shape
    (opt) z_step: If we have n possible z-shape turing points, (n = abs(pin.x - parent_pin.x) + abs(pin.y - parent_pin.y) - 2),
                    then we pick turning point every z_step, which will generate int(n/(z_step)) z-shape routing candidates
    (opt) max_z: int, the maximum z-shape routing candidates we want to generate (if we have more than max_z candidates, we will increase the z_step until we have less than max_z candidates)
    (opt) c_step: int, the step of c-shape routing candidates
    (opt) max_c: int, the maximum c-shape routing candidates we want to generate
    (opt) max_c_out_ratio: float, when pick c turning points, we need to extend the edge, this is the maximum ratio that we extend the edge
    
return: list of routing candidates (two l-shape routing)
        for each routing candidates, we store following routing metrics
        2. path_length (wirelength)
        3. via_map: numpy array, (2,n), n is the number of turning points, NOTE: the via_map is ordered from parent pin to child pin
        4. edge_info tuple, edge is the gcell edge
            encoding_matrix: index tensor L = (2, edge_num), note that the edge should be ordered,  
                THE FIRST EDGE ALWAYS CONNECTS TO THE CHILD PIN
                THE LAST EDGE ALWAYS CONNECTS TO THE PARENT PIN
            is_hor, tensor (edge_num), bool, whether the edge is horizontal
        5. seg_info repeats: tensor, (segment), each element is the number of the edge of the segment
"""
def two_pin_routing(pin, parent_pin, xmax, ymax, pattern_level=2, z_step=4, max_z=10, c_step = 2, max_c= 10, max_c_out_ratio = 1):
    x_width = abs(pin.x - parent_pin.x)
    y_width = abs(pin.y - parent_pin.y)
    if pin.x == parent_pin.x:
        # 2. path_length
        path_length = y_width
        # 3. turning_point
        turning_point = np.empty((2,0))
        # 4. edge_info, edge_idx: (2, edge_num), the (x, y) index of the edge
        edge_idx, is_hor, edge_num = get_edge_info(pin.x, pin.x, pin.y, parent_pin.y)
        return [(path_length, turning_point, (edge_idx, is_hor), np.array([edge_num]))]
    elif pin.y == parent_pin.y:
        # vertical
        # 2. path_length
        path_length = x_width
        # 3. turning_point
        turning_point = np.empty((2,0))
        # 4. edge_info
        edge_idx, is_hor, edge_num = get_edge_info(pin.x, parent_pin.x, pin.y, pin.y)
        return [(path_length, turning_point, (edge_idx, is_hor), np.array([edge_num]))]
    else:
        result = []
        # 1. L-shape routing, opt1 goes through child's y first, opt2 goes through parent's y first
        # 1.2. path_length
        path_length = x_width + y_width
        # 1.3. turning_point
        turning_point1 = np.array([[parent_pin.x],[pin.y]])
        turning_point2 = np.array([[pin.x],[parent_pin.y]])
        # 1.4. edge_info
        # opt 1 child y not change + parent x not change
        # child y part (step 1, s1)
        edges1,is_hor1, edge_num1 = get_edge_info(pin.x, parent_pin.x, pin.y, pin.y)
        # parent x part (step 2, s2)
        edges2,is_hor2, edge_num2 = get_edge_info(parent_pin.x, parent_pin.x, pin.y, parent_pin.y)
        edge_idx1 = np.concatenate((edges1, edges2), axis = 1)
        is_hor1_s1 = np.concatenate((is_hor1, is_hor2))
        edge_num1 = np.array((edge_num1, edge_num2))
        # opt 2 parent y + child x
        # child x part (step 1, s1)
        edges1,is_hor1, edge_num1 = get_edge_info(pin.x, pin.x, pin.y, parent_pin.y)
        # parent y part (step 2, s2)
        edges2,is_hor2, edge_num2 = get_edge_info(pin.x, parent_pin.x, parent_pin.y, parent_pin.y)
        result += [(path_length, turning_point1, (edge_idx1, is_hor1_s1), edge_num1),
                   (path_length, turning_point2, (np.concatenate((edges1, edges2), axis = 1), np.concatenate((is_hor1, is_hor2))), np.array((edge_num1, edge_num2)))]

        # 2. Z-shape routing
        if pattern_level >= 2:
            num_z = int((x_width - 1 + y_width - 1) / z_step)
            if num_z > max_z:
                z_step = math.ceil((x_width - 1 + y_width - 1) / max_z)
                num_z = int((x_width - 1 + y_width - 1) / z_step)
            if num_z > 0:
                for z_index in range(num_z):
                    turning_x = min(pin.x, parent_pin.x) + (z_index + 1) * z_step
                    if turning_x >= max(pin.x, parent_pin.x):
                        turning_x = min(pin.x, parent_pin.x)
                        turning_y = min(pin.y, parent_pin.y) + (z_index + 1) * z_step - (
                                    max(pin.x, parent_pin.x) - min(pin.x, parent_pin.x) - 1)
                        assert turning_y < max(pin.y, parent_pin.y)
                    else:
                        turning_y = min(pin.y, parent_pin.y)
                    single_z_result = z_pattern_routing(pin, parent_pin, (turning_x, turning_y), xmax, ymax)
                    result.append(single_z_result)
        
    # C routing, even for straight line, we have to consider C routing
    # for C routing, we have four options. 1. x = minx, y < miny; 2. x = minx, y > maxy; 3. x < minx, y = miny, 4. x > maxx, y = miny
    if pattern_level >= 3:
        # the maximum extend width along x axis (negative direction)
        extend_x_min = min(x_width * max_c_out_ratio, min(pin.x, parent_pin.x)) 
        # the maximum extend width along x axis (positive direction)
        extend_x_max = min(x_width * max_c_out_ratio, xmax - max(pin.x, parent_pin.x) - 1)
        # the maximum extend width along y axis (negative direction)
        extend_y_min = min(y_width * max_c_out_ratio, min(pin.y, parent_pin.y))
        # the maximum extend width along y axis (positive direction)
        extend_y_max = min(y_width * max_c_out_ratio, ymax - max(pin.y, parent_pin.y) - 1)
        total_extend_length = extend_x_min + extend_x_max + extend_y_min + extend_y_max

        num_c = int(total_extend_length / c_step)
        if num_c > max_c:
            c_step = math.ceil(total_extend_length / max_c)
            num_c = int(total_extend_length / c_step)
        random_offset = random.randint(1, c_step)
        if num_c > 0:
            for c_index in range(num_c):
                # start from opt1 ((c_index) * c_step + random_offset) <= extend_y_min
                if ((c_index) * c_step + random_offset) <= extend_y_min:
                    turning_x = min(pin.x, parent_pin.x)
                    turning_y = min(pin.y, parent_pin.y) - ((c_index) * c_step + random_offset)
                    assert turning_y >= 0
                # opt2 extend_y_min < ((c_index) * c_step + random_offset) <= extend_y_min + extend_y_max
                elif ((c_index) * c_step + random_offset) <= extend_y_min + extend_y_max:
                    turning_x = min(pin.x, parent_pin.x)
                    turning_y = max(pin.y, parent_pin.y) + ((c_index) * c_step + random_offset) - extend_y_min
                    assert turning_y < ymax
                # opt3 extend_y_min + extend_x_min < ((c_index) * c_step + random_offset) <= extend_y_min + extend_x_min + extend_y_max
                elif ((c_index) * c_step + random_offset) <= extend_y_min + extend_x_min + extend_y_max:
                    turning_x = min(pin.x, parent_pin.x) - ((c_index) * c_step + random_offset) + extend_y_min + extend_y_max
                    turning_y = min(pin.y, parent_pin.y)
                    assert turning_x >= 0
                # opt4 extend_y_min + extend_x_min + extend_y_max < ((c_index) * c_step + random_offset) <= extend_y_min + extend_x_min + extend_y_max + extend_x_max
                elif ((c_index) * c_step + random_offset) <= extend_y_min + extend_x_min + extend_y_max + extend_x_max:
                    turning_x = max(pin.x, parent_pin.x) + ((c_index) * c_step + random_offset) - extend_y_min - extend_x_min - extend_y_max
                    turning_y = min(pin.y, parent_pin.y)
                    assert turning_x < xmax
                else:
                    assert False
                single_c_result = z_pattern_routing(pin, parent_pin, (turning_x, turning_y), xmax, ymax)
                result.append(single_c_result)
    return result

"""
Initialize the candidate pool for each net. Here, we only consider l-shape routing
Require:
    nets: list of Net objects
    xmax: int, x max of the routing region
    ymax: int, y max of the routing region
    pattern_level: int, the level of pattern routing: 1 only l-shape, 2 add z shape 3 add monotonic shape
    max_z: int, the maximum number of z shape routing candidates
return:
    candidate_pool: list of candidate pool for each net
        candidate_pool[i][j][k][1][m] is the i_th net, j_th topology, k_th 2-pin net (connects pinID k to its parent pin), m_th candidate
                      [i][j][k][0] is the (pin, parent_pin) tuple
            it includes following metrics
            2. path_length (wirelength)
            3. turning_point (via count)
            4. edge_info tuple, edge is the gcell edge
                encoding_matrix: index tensor (2, edge_num), note that the edge should be ordered, 
                    THE FIRST EDGE ALWAYS CONNECTS TO THE CHILD PIN
                    THE LAST EDGE ALWAYS CONNECTS TO THE PARENT PIN
                is_hor, tensor (edge_num), bool, whether the edge is horizontal
            5. seg_info repeats: tensor, (segment), each element is the number of the edge of the segment

"""
def get_initial_candidate_pool(nets: List[Net],xmax: int, ymax: int, device: str, pattern_level = 2, max_z = 10, z_step = 4,c_step = 2, max_c= 10, max_c_out_ratio = 1):
    candidate_pool = []
    for netIdx,net in enumerate(nets):
        candidate_pool.append([])
        num_tree = len(net.pins)
        for tree in range(num_tree):
            candidate_pool[-1].append([])
            for pin in net.pins[tree]:
                candidate_pool[-1][-1].append([])
                # if the pin is the root pin, then no need to consider routing
                if pin.parent_pin == -1 or pin.parent_pin == None:
                    candidate_pool[-1][-1][-1] = []
                    continue
                candidate_pool[-1][-1][-1] = [(pin,net.pins[tree][pin.parent_pin]),two_pin_routing(pin, net.pins[tree][pin.parent_pin], xmax, ymax, z_step = z_step,pattern_level = pattern_level, max_z = max_z, c_step = c_step, max_c= max_c, max_c_out_ratio = max_c_out_ratio)]
    return candidate_pool

"""
Visualize the result, for a xmax*ymax routing region, we need to visualize both pins and routing paths
therefore, a result (2xmax - 1, 2ymax - 1) matrix is needed.
Then, result[2i+1,2j] = hor_path[i,j], result[2i,2j+1] = ver_path[i,j] 
Require:
    hor_path: sparse tensor, (xmax* ymax - 1,n), n is the number of candidates
    ver_path: sparse tensor, (xmax - 1*ymax, n), n is the number of candidates
    p: tensor, (n), p[i] is the probability of the i_th candidate to be selected
"""
def visualize_result(xmax,ymax,hor_path, ver_path, p,name = None, capacity_mat = None, caption = 'plot'):
    result = torch.zeros((2*xmax-1, 2*ymax-1)).to(hor_path.device)
    result[::2,1::2] = torch.matmul(hor_path,p).to_dense().reshape((xmax,ymax-1))
    result[1::2,::2] = torch.matmul(ver_path,p).to_dense().reshape((xmax-1,ymax))

    if capacity_mat is not None:
        ratio_result = torch.zeros((2*xmax-1, 2*ymax-1)).to(hor_path.device)
        # result[::2,1::2] element-wise division capacity_mat[0] (show the ratio rather than value)
        ratio_result[::2,1::2] = torch.div(result[::2,1::2], capacity_mat[0])
        # result[1::2,::2] element-wise division capacity_mat[1] (show the ratio rather than value)
        ratio_result[1::2,::2] = torch.div(result[1::2,::2], capacity_mat[1])

    # traslate from edge to gcell heatmap
    real_result= torch.zeros((xmax, ymax)).to(hor_path.device)
    # real_result[i,j] = (result[2i+1,2j] + result[2i,2j+1] + result[2i-1,2j] + result[2i,2j-1])/4
    real_result[1:-1,1:-1] = (result[2:-2:2,3:-1:2] + result[3:-1:2,2:-2:2] + result[1:-3:2,2:-2:2] + result[2:-2:2,1:-3:2])/4
    # for boundary, we only consider the 3 (or 2) directions, and divide by 3 (or 2)
    real_result[0,1:-1] = (result[1,2:-2:2] + result[0,3:-1:2] + result[0,1:-3:2])/3
    real_result[-1,1:-1] = (result[-2,2:-2:2] + result[-1,3:-1:2] + result[-1,1:-3:2])/3
    real_result[1:-1,0] = (result[2:-2:2,1] + result[3:-1:2,0] + result[1:-3:2,0])/3
    real_result[1:-1,-1] = (result[2:-2:2,-2] + result[3:-1:2,-1] + result[1:-3:2,-1])/3
    real_result[0,0] = (result[1,0] + result[0,1])/2
    real_result[0,-1] = (result[1,-1] + result[0,-2])/2
    real_result[-1,0] = (result[-2,0] + result[-1,1])/2
    real_result[-1,-1] = (result[-2,-1] + result[-1,-2])/2

    result = real_result
    ratio_result = result / (capacity_mat[0].mean() + capacity_mat[1].mean()) * 2
    # import seaborn as sns

    # visualize the numpy result with barm, with caption as title
    ax = sns.heatmap(result.cpu().detach().numpy())
    ax.set_title(caption)
    # show the result
    if name is None:
        name = 'result'
    # save the result into './figs/${name}.png'
    # mkdir figs if not exist
    import os
    if not os.path.exists('./figs'):
        os.mkdir('./figs')
    plt.savefig('./figs/{}.png'.format(name))
    plt.close()
    if capacity_mat is not None:
        ax = sns.heatmap(ratio_result.cpu().detach().numpy())
        ax.set_title(caption)
        plt.savefig('./figs/{}_ratio.png'.format(name))
        plt.close()
    return

"""
visualize distribution of probability p
"""
def visualize_p(p,name,base = 10):
    plt.figure()
    plt.hist(p, bins=100)
    plt.xlabel("max p value")
    plt.ylabel("count")
    # yscale and base is base
    plt.yscale('log',base=base)
    plt.savefig("./figs/" + name + ".png")
    plt.close()

# main
if __name__ == '__main__':
    # test z_pattern_routing()
    test_pin = Pin(3,4)
    test_pin2 = Pin(5,2)
    result = two_pin_routing(test_pin, test_pin2, 10, 10, z_step = 2)
    turning_point = (3,1)
    result = z_pattern_routing(test_pin, test_pin2,  turning_point,10, 10)

    print(result)



"""
Get the via cost label.
Require:
    hor_L: # of horizontal layers
    ver_L: # of vertical layers
    hor_first: if True, the first layer is horizontal, else the first layer is vertical
Return:
    via_cost_label: a tuple (non_turning_via_cost, turning_via_cost), the cost of via for turning/nonturning edges.
        non_turning_via_cost: (L * L): non_turning_via_cost[i,j] is the via cost when the edge is assigned to layer i and the next edge is assigned to layer j, L is the max(hor_L, ver_L)
        turning_via_cost: (hor_L * ver_L): turning_via_cost[i,j] is the via cost when the edge is assigned to hor layer i and the next edge is assigned to ver layer j
"""
def get_via_cost_label(hor_L, ver_L, hor_first,device):
    L = max(hor_L, ver_L)
    non_turning_via_cost = torch.zeros((L, L)).to(device).float()
    turning_via_cost = torch.zeros((L, L)).to(device).float()
    # for non_turning_via_cost, non_turning_via_cost[i,j] = 2 * abs(i - j)
    for i in range(L):
        for j in range(L):
            non_turning_via_cost[i,j] = 2 * abs(i - j)
    # for turning_via_cost, 
    # if hor_first, 
    # turning_via_cost[i,j] = abs(2 * (i - j - 1) + 1 ) if i>= j else abs(2 * (j - i) + 1)
    # else,
    # turning_via_cost[i,j] = abs(2 * (i - j) + 1 ) if i>= j else abs(2 * (j - i - 1) + 1)   
    if hor_first:
        for i in range(L):
            for j in range(L):
                turning_via_cost[i,j] = abs(2 * (i - j - 1) + 1 ) if i>= j else abs(2 * (j - i) + 1)
    else:
        for i in range(L):
            for j in range(L):
                turning_via_cost[i,j] = abs(2 * (i - j) + 1 ) if i>= j else abs(2 * (j - i - 1) + 1)
    return non_turning_via_cost, turning_via_cost


"""
Save path result into ./2D_result/${name}.pt
Require:
    nets: list of Net, the nets to be routed
    selected_index: tensor, (n), selected_index[i] is the index of the i_th candidate to be selected
    name: string, the name of the saved file
    candidate_pool: list of candidate pool for each net, each candidate is a 2-pin net from child to its parent (therefore, for root pin, there is no candidate since it has no parent)
    p_index2pattern_index: List((int,int,int,int)), the map from index in probability array to the index in candidate_pool
Save:
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
"""
def save_path_result(nets, selected_index,candidate_pool,p_index2pattern_index,name):
    selected_candidates = p_index2pattern_index[selected_index.cpu()]
    edge_path = []
    is_hor_list = []
    end_index = []
    current_edge_index = 0
    start_idx = []
    net_start_idx = {}
    net_end_idx = {}
    pin2edge_idx = {} # map from (netname, child_pinidx) to corresponding start and end edge index
    pinID2Idx = {} # map from ((netidx, treeidx, pinidx) to idx in start_idx/end_idx), if the pin is a root, then there is no entry in this map
    last_net_idx = -1
    for pin_idx, idx in enumerate(selected_candidates):
        # idx is (netidx, treeidx, pinidx (child pin idx), candidateidx)
        if idx[0] != last_net_idx:
            net_start_idx[nets[idx[0]].net_name] = current_edge_index
            if last_net_idx != -1:
                net_end_idx[nets[last_net_idx].net_name] = current_edge_index
            last_net_idx = idx[0]
        candidate = candidate_pool[idx[0]][idx[1]][idx[2]][idx[3]]
        edge_path.append(candidate[2][0])
        is_hor_list.append(candidate[2][1])
        end_index.append(current_edge_index + candidate[2][0].shape[1] -  1)
        start_idx.append(current_edge_index)
        current_edge_index += candidate[2][0].shape[1]
        pinID2Idx[(idx[0], idx[1], idx[2])] = pin_idx
        pin2edge_idx[(nets[idx[0]].net_name,idx[2])] = (start_idx[-1], end_index[-1])
    net_end_idx[nets[last_net_idx].net_name] = current_edge_index
    # then for each parent pin, we are concerned about the via cost among different 2-pin nets.
    # edge_idx1 (E') stores the edge index in edge_list for the head 
    # edge_idx2 (E') stores the edge index in edge_list for the tail
    # weights (E') (the weight for via cost, heuritic method)
    edge_idx1 = []
    edge_idx2 = []
    weights = []
    is_steiner = []
    pin_max = [] # max layer inside this gcell 
    pin_min = [] # min layer inside this gcell
    edge_path = np.concatenate(edge_path, axis = 1) # (2,E)
    is_hor_list = np.concatenate(is_hor_list, axis = 0) # (E)
    # scatter_index = np.ones_like(is_hor_list) * len(start_idx) 
    scatter_index = np.ones_like(is_hor_list) * len(start_idx)
    for netID, net in enumerate(nets):
        num_tree = net.num_tree
        for treeID in range(num_tree):
            for pinID, pin in enumerate(net.pins[treeID]):
                # if the pin is root, we also skip
                if (netID,treeID,pinID) not in pinID2Idx.keys():
                    continue
                if pin.is_steiner == 1:
                    # for steiner pins, the physical pin layer should be empty
                    assert(len(pin.physical_pin_layers) == 0)
                    is_steiner.append(True)
                    pin_max.append(0)
                    pin_min.append(1e9)
                else:
                    is_steiner.append(False)
                    pin_max.append(np.max(pin.physical_pin_layers))
                    pin_min.append(np.min(pin.physical_pin_layers))
                edge_list = [] # edge_list temporarily stores all related edge
                edge_list.append(start_idx[pinID2Idx[(netID,treeID,pinID)]]) # for parent node, the start edge connects to children
                scatter_index[start_idx[pinID2Idx[(netID,treeID,pinID)]]] = pinID2Idx[(netID,treeID,pinID)] # set the edge's scatter index as pin index
                if pin.child_pins is None:
                    continue
                for child_pin in pin.child_pins:
                    edge_list.append(end_index[pinID2Idx[(netID,treeID,child_pin)]]) # for parent node, the end edge connects to children
                    scatter_index[end_index[pinID2Idx[(netID,treeID,child_pin)]]] = pinID2Idx[(netID,treeID,pinID)] 
                if len(edge_list) == 2:
                    weights.append(1)
                    edge_idx1.append(edge_list[0])
                    edge_idx2.append(edge_list[1])
                elif len(edge_list) == 3:
                    weights+= [0.5 for i in range(3)]
                    edge_idx1 += edge_list # 0,1,2
                    edge_idx2 += [edge_list[1],edge_list[2],edge_list[0]] # 1,2,0
                elif len(edge_list) == 4:
                    weights += [0.35 for i in range(6)]
                    edge_idx1 += edge_list 
                    edge_idx1 += [edge_list[0],edge_list[1]]# 0,1,2,3,0,1
                    edge_idx2 += [edge_list[1],edge_list[2],edge_list[3],edge_list[0], edge_list[2], edge_list[3]] # 1,2,3,0,2,3
                elif len(edge_list) == 5:
                    weights += [0.2 for i in range(10)]
                    edge_idx1 += [edge_list[0], edge_list[1], edge_list[2], edge_list[3], edge_list[4], edge_list[0], edge_list[1], edge_list[2], edge_list[0], edge_list[1]]# 0,1,2,3,4,0,1,2,0,1
                    edge_idx2 += [edge_list[1], edge_list[2], edge_list[3], edge_list[4], edge_list[0], edge_list[2], edge_list[3], edge_list[4], edge_list[3], edge_list[4]] # 1,2,3,4,0,2,3,4,3,4
                else:
                    raise NotImplementedError("Not implemented for more than 5 pins")

    

    edge_idx1 = np.array(edge_idx1)
    edge_idx2 = np.array(edge_idx2)
    weights = np.array(weights)
    end_index = np.array(end_index)
    start_idx = np.array(start_idx)
    is_steiner = np.array(is_steiner)
    # save to file
    # mkdir 2D_result if not exist
    import os
    if not os.path.exists("./2D_result"):
        os.mkdir("./2D_result")
    torch.save((pin2edge_idx,net_start_idx,net_end_idx, edge_path, is_hor_list, end_index, start_idx, edge_idx1, edge_idx2, weights,is_steiner, scatter_index, pin_max, pin_min), "./2D_result/{}.pt".format(name))

"""
x_step_size: the size of each step in x direction
x_step_count: the number of steps in x direction. For example, when x_step_count = [2,1] and x_step_size = [3000,4000], then the gcell size for each gcell is [3000,3000,4000], and the first output (xmin) is [0,3000,6000], the second output (xmax) is [3000,6000,10000]
y_step_size: the size of each step in y direction
y_step_count: the number of steps in y direction
"""
def step2coordinates(x_step_size,x_step_count,y_step_size,y_step_count):
    x_step =np.repeat(x_step_size,x_step_count)
    y_step =np.repeat(y_step_size,y_step_count)
    x_min = np.cumsum(x_step) - x_step
    x_max = np.cumsum(x_step)
    y_min = np.cumsum(y_step) - y_step
    y_max = np.cumsum(y_step) 
    return x_min, x_max, y_min, y_max



"""
Output guide files
the file is organized as follows:
netname
(
x1 y1 x2 y2 layer_name
x1 y1 x2 y2 layer_name
...
)
netname2
(
...
)
Require:
    cap: a tuple (hor_cap, ver_cap), the capacity of horizontal and vertical layers
        hor_cap: (hor_L, xmax * ymax - 1): hor_cap[i] is the capacity of horizontal layer i
        ver_cap: (ver_L, xmax - 1 * ymax): dir_cap[i] is the capacity of vertical layer i
    file_path: the path to save the guide file
    nets: the nets, list(Net)
    edge_path: [2,E]
    net_start_idx: the start index of each net in the selected_edge_layer, dict(net_name: start_idx)
    net_end_idx: the end index of each net in the selected_edge_layer, dict(net_name: end_idx): NOTE: it is the same with the start idx of next net
    selected_edge_layer: the selected edge layer, E, the value is from 0 to L-1
    is_hor: the is_hor of each edge, E

    07/09 We implemented patch-based post-process.
    (Another idea: for pins in boundary, also add neighbor gcells.)
    1. If the gcell includes pins. Then expand pin for up 2 layers and down 2 layers. And expand to 3*3 for each layer
    2. If the edge is very long, add upper/lower gcells when it is out of the threshold
    3. If Gcell has inevitable violations, G-cells on the two sides of the G-cell with violation, along with the three G-cells above are patched.
"""
def write_guide(cap,routingregion,file_path, nets, edge_path, net_start_idx, net_end_idx, selected_edge_layer, is_hor, hor_first,x_min, x_max, y_min, y_max):
    
    # below parameters are from CUGR2.0 and are for 
    wire_patch_threshold = 2
    wire_patch_inflation_rate = 1.2
    pin_patch_threshold = 20
    effective_wire_patch_threshold = wire_patch_threshold
    
    edge_path = edge_path.cpu().numpy()
    actual_layer = selected_edge_layer * 2 + 1 + (is_hor != hor_first).cpu().int().numpy() + (is_hor == hor_first).cpu().int().numpy() * 2
    # print the histogram of the actual layer
    print("actual layer hist: {}".format(np.histogram(actual_layer,bins=range(2,actual_layer.max()+1))[0]))
    full_cap = np.zeros((x_max.shape[0],y_max.shape[0],cap[0].shape[-1] + cap[1].shape[-1])) # [x,y,layer],encoding the left capacity of each gcell
    for i in range(full_cap.shape[-1]):
        if i % 2 == 0 and hor_first is False: # hor_first is False, means the second layer is horizontal. Here, we start from the second layer
            full_cap[:,:-1,i] = cap[0][:,:,i//2].cpu()
        else:
            full_cap[:-1,:,i] = cap[1][:,:,i//2].cpu()
    
    # minus the used capacity by edge_path
    full_sparse = full_cap.copy()
    for i in range(edge_path.shape[-1]):
        full_sparse[edge_path[0,i],edge_path[1,i],actual_layer[i]-2] -= 1
    
    # print avg sparse of each layer
    for layer_idx in range(full_sparse.shape[-1]):
        print("total / avg / min ava. track of layer {}: {}, {}, {}".format(layer_idx+2,np.mean(full_cap[:,:,layer_idx]),np.mean(full_sparse[:,:,layer_idx]),np.min(full_sparse[:,:,layer_idx])))
    
    # write netname at first
    num_layer = len(routingregion.cap_mat_3D[0]) + len(routingregion.cap_mat_3D[1]) + 1 # +1 because metal 1 is not used
    num_1_gcell_net = 0
    pin_distribution = {} # record the distribution of pins in each gcell. key: num of pins in gcell, value: num of gcells
    pin_layer_distribution = {} # record the distribution of pin layers. key: layer, value: num of pins
    with open(file_path,'w+') as f:
        for net in nets:
            f.write(net.net_name)
            f.write('\n')
            f.write('(\n')
            # if net.need_route is False, (only one gcell) directly output based on gcell_list
            if net.need_route is False:
                num_1_gcell_net += 1
                min_gcell_layer = min([gcell[2] for gcell in net.gcell_list])
                max_gcell_layer = max([gcell[2] for gcell in net.gcell_list])
                gcell_x = net.gcell_list[0][0] # gcell_x is the same for all gcells
                gcell_y = net.gcell_list[0][1]
                for layer in range(min_gcell_layer, max_gcell_layer + 1):
                    write(f,gcell_x,gcell_y,x_min,x_max, y_min,y_max,layer+ 1)
            else:
                start_idx = net_start_idx[net.net_name]
                end_idx = net_end_idx[net.net_name]
                # we use gcell_loc to track the selected gcell location, finally, for each pin, we will start from its bottom layer to top layer
                gcell_loc = {}
                for i in range(start_idx, end_idx):
                    write(f,edge_path[0,i],edge_path[1,i], x_min, x_max, y_min, y_max, actual_layer[i])
                    gcell_loc[(edge_path[0,i],edge_path[1,i],actual_layer[i])] = True
                    write(f,edge_path[0,i] + 1 - int(is_hor[i]),edge_path[1,i] + int(is_hor[i]), x_min, x_max, y_min, y_max, actual_layer[i])
                    gcell_loc[(edge_path[0,i] + 1 - int(is_hor[i]),edge_path[1,i] + int(is_hor[i]),actual_layer[i])] = True
                    # WIRE PATCH: if the wire gcell is not sparse ( < 2), we add the upper and lower gcells (unless they are too crowded)
                    if full_sparse[edge_path[0,i],edge_path[1,i],actual_layer[i]-2] < effective_wire_patch_threshold:
                        # add upper gcell
                        if actual_layer[i] -2 + 1 < full_sparse.shape[-1] and full_sparse[edge_path[0,i],edge_path[1,i],actual_layer[i]-1] < effective_wire_patch_threshold:
                            write(f,edge_path[0,i],edge_path[1,i], x_min, x_max, y_min, y_max, actual_layer[i]+1)

                        # add lower gcell
                        if actual_layer[i]-2 > 0 and full_sparse[edge_path[0,i],edge_path[1,i],actual_layer[i]-3] < effective_wire_patch_threshold:
                            write(f,edge_path[0,i],edge_path[1,i], x_min, x_max, y_min, y_max, actual_layer[i]-1)
                        effective_wire_patch_threshold = wire_patch_threshold
                    else:
                        # otherse, inflate the threshold
                        effective_wire_patch_threshold = effective_wire_patch_threshold * wire_patch_inflation_rate

                # now, we do not have multiple tree candidates
                assert len(net.pins) == 1
                # for each pin object
                for pin in net.pins[0]:
                    # for each non-steiner pin, we need to output the gcell location from bottom layer to top layer
                    
                    if pin.is_steiner == False:
                        smallest_layer = min(pin.physical_pin_layers)
                        largest_layer = max(pin.physical_pin_layers)
                        pin_distribution[int(len(pin.physical_pin_layers))] = pin_distribution.get(int(len(pin.physical_pin_layers)),0) + 1
                        for layer in pin.physical_pin_layers:
                            pin_layer_distribution[int(layer)] = pin_layer_distribution.get(int(layer),0) + 1
                        
                        smallest_wire_layer = 1e9 
                        largest_wire_layer = -1e9
                        # calculate smallest_wire_layer, which is actual layer idx
                        for wire_layer_idx in range(2,num_layer+1):
                            if (pin.x,pin.y,wire_layer_idx) in gcell_loc.keys():
                                smallest_wire_layer = wire_layer_idx
                                break
                        # calculate largest_wire_layer, also actual layer idx (start from metal1, )
                        for wire_layer_idx in range(num_layer,1,-1):
                            if (pin.x,pin.y,wire_layer_idx) in gcell_loc.keys():
                                largest_wire_layer = wire_layer_idx
                                break

                        assert smallest_wire_layer != 1e9, "the wire must be in the gcell"
                        assert largest_wire_layer != -1e9, "the wire must be in the gcell"

                        start_layer = min(smallest_layer+1,smallest_wire_layer)
                        end_layer = max(largest_layer+1,largest_wire_layer)
                        for layer_idx in range(start_layer,end_layer+1):
                            write_one_layer(f,pin.x, pin.y, x_min, x_max, y_min, y_max,layer_idx)
                            
                        # PIN PATCH, for each pin at layer 0, if its sparcity score is below 20 (pin_patch_threshold in CUGR 2.0), we exapnd it to 3*3
                        if smallest_layer == 0:
                            if full_sparse[pin.x,pin.y,1] < pin_patch_threshold or full_sparse[pin.x,pin.y,2] < pin_patch_threshold:
                                for i in range(-1,2):
                                    for j in range(-1,2):
                                        if pin.x + i >= 0 and pin.x + i < len(x_max) and pin.y + j >= 0 and pin.y + j < len(y_max):
                                            write_one_layer(f,pin.x + i, pin.y + j, x_min, x_max, y_min, y_max,1)
                                            write_one_layer(f,pin.x + i, pin.y + j, x_min, x_max, y_min, y_max,2)
                                            write_one_layer(f,pin.x + i, pin.y + j, x_min, x_max, y_min, y_max,3)
                                        
                # check each gcell location
                visited = {}
                for gcell in gcell_loc.keys():
                    # if the x,y not visied, we traverse from bottom to top, to get min layer, and then get max layer.
                    # and then add via for each middle layer
                    if (gcell[0],gcell[1]) not in visited.keys():
                        for layer_idx in range(1,num_layer + 1):
                            if (gcell[0],gcell[1],layer_idx) in gcell_loc.keys():
                                min_layer = layer_idx
                                break
                        for layer_idx in range(num_layer,0,-1):
                            if (gcell[0],gcell[1],layer_idx) in gcell_loc.keys():
                                max_layer = layer_idx
                                break
                        for layer_idx in range(min_layer+1,max_layer):
                            write_one_layer(f,gcell[0], gcell[1], x_min, x_max, y_min, y_max,layer_idx)
                        visited[(gcell[0],gcell[1])] = True
                
                # padding, we need the sparse info
                # padding one, 
            f.write(')\n')
    print('num_1_gcell_net: ',num_1_gcell_net)
    print('pin_distribution: ',pin_distribution)
    print('pin_layer_distribution: ',pin_layer_distribution)
    
def write_one_layer(f,x,y,x_min,x_max, y_min,y_max,layer_idx):
    layer_name = 'Metal' + str(layer_idx)
    f.write(str(x_min[x]))
    f.write(' ')
    f.write(str(y_min[y]))
    f.write(' ')
    f.write(str(x_max[x]))
    f.write(' ')
    f.write(str(y_max[y]))
    f.write(' ')
    f.write(layer_name)
    f.write('\n')

def write(f,x,y,x_min,x_max, y_min,y_max,this_layer,total_num_layer = 9, range_num = 1):
    #selected_edge_layer[i] * 2 + 1 + int(is_hor[i]) - int(hor_first)*(1- 2*int(is_hor[i]))
    start_layer = this_layer
    range_num = 3 if start_layer == 1 else range_num
    for i in range(range_num):
        write_one_layer(f,x,y,x_min,x_max, y_min,y_max,start_layer + i)

"""
To better estimate the overflow cost, we need to set the influence by each physical pin.
NOTE: steiner points here are not considered, instead, it is related with tree stucture.
Require:
    layer_info: the layer information, read from CUGR2
Return:
    hor_pin_demand: (xmax, ymax - 1)
    ver_pin_demand: (xmax - 1, ymax)
"""
def get_pin_demand(nets, args, layer_info):
    hor_pin_demand = np.zeros((args.xmax,args.ymax - 1))
    ver_pin_demand = np.zeros((args.xmax - 1,args.ymax))
    layer2number = {} # layer number : number of this case
    for net in nets:
        for pin in net.pins[0]: # physical pins for all trees are the same. So we only need pick net.pins[0] (the first tree candidate)
            if pin.is_steiner is not True:
                for layer_idx in range(pin.physical_pin_layers[0],pin.physical_pin_layers[1] + 1):
                    real_layer_idx = layer_idx - 1 # Metal layer 0 is not used in routing.
                    if real_layer_idx < 0:
                        continue
                    
                this_density = pin.physical_pin_layers[1] - pin.physical_pin_layers[0] + 1
                if this_density not in layer2number.keys():
                    layer2number[this_density] = 1
                else:
                    layer2number[this_density] += 1
                assert(this_density >= 0)
    print("layer_number statistics: {}".format(layer2number))
    return hor_pin_demand, ver_pin_demand


"""
update capacity based on pin_density
"""
def upd_cap_by_pin(cap_mat, pin_density):
    hor_pin_density = (pin_density[:,:-1] + pin_density[:,1:])/2  # /2 because the pin density actually only influence half layers (half is horizontal/vertical)
    ver_pin_density = (pin_density[:-1,:] + pin_density[1:,:])/2
    cap_mat[0] = cap_mat[0] - torch.tensor(hor_pin_density)
    cap_mat[1] = cap_mat[1] - torch.tensor(ver_pin_density)
    return cap_mat
    

"""
Write the input file for CUGR after we select the concrete paths
select_threshold: Set a Probability threshold: t, select candidates from lager p to smaller p, until sum of candidate probabilities are larger than t
"""
def write_CUGR_input(nets, p,p_index_full,candidate_pool,p_index2pattern_index,name, select_threshold):
    max_p, selected_index = scatter_max(p,p_index_full)
    total_p = torch.zeros_like(max_p).to(max_p.device)
    last_selected_index = torch.clone(selected_index)
    remaining_p = torch.clone(p)
    remaining_p_index_full = torch.clone(p_index_full)
    remaining_p_index2pattern_index = p_index2pattern_index.copy()
    final_selected_pattern_idx = []
    # selected_index_full = one hot of selected_index
    # while not all elements in total_p is larger than select_threshold, we select more candidates
    while (total_p >= (select_threshold - 1e-6)).all().cpu().item() is False:
        # selected_index_full means those selected in the last iteration
        total_p += max_p
        print(" # of selected index: ", len(last_selected_index))
        selected_index_full = torch.zeros_like(remaining_p_index_full)
        selected_index_full[last_selected_index] = 1
        # selected_index_full == 0 means the remaining not selected candidates
        remaining_p = remaining_p[selected_index_full == 0]
        remaining_p_index_full = remaining_p_index_full[selected_index_full == 0]
        if len(last_selected_index) == 1:
            final_selected_pattern_idx.append(remaining_p_index2pattern_index[last_selected_index.cpu()].reshape((1,-1)))
        else:
            final_selected_pattern_idx.append(remaining_p_index2pattern_index[last_selected_index.cpu()])
        remaining_p_index2pattern_index = remaining_p_index2pattern_index[(selected_index_full == 0).cpu()]
        max_p, last_selected_index = scatter_max(remaining_p,remaining_p_index_full,dim_size = selected_index.shape[0])
        last_selected_index = last_selected_index[total_p < (select_threshold - 1e-6)]
        # max_p = 0 if total_p >= select_threshold
        max_p = torch.where(total_p >= (select_threshold - 1e-6), torch.zeros_like(max_p), max_p).to(p.device)
        
    selected_candidates = np.concatenate(final_selected_pattern_idx)
    # selected_candidates = p_index2pattern_index # used for debug (output all candidates)
    with open('./CUGR2_guide/CUgr_'+name+'.txt','w') as f:
        for pin_idx, idx in enumerate(selected_candidates):
            # idx is (netidx, treeidx, pinidx (child pin idx), candidateidx)
            # skip if idx is [0,0,0,0]
            if (idx == 0).all():
                continue
            # write nets[idx[0]].net_name, nets[idx[0]].net_ID, idx[2](child pin idx)
            f.writelines([nets[idx[0]].net_name, ' ', str(nets[idx[0]].net_ID), ' ', str(idx[2])])
            f.write('\n')
            # write the path
            candidate = candidate_pool[idx[0]][idx[1]][idx[2]][1][idx[3]]
            via_map = candidate[1]
            # range in reverse order, because the via_map is from child pin to parent pin, but CUGR expects the opposite order
            for via_idx in range(via_map.shape[1]-1,-1,-1):
                f.writelines([str(via_map[0,via_idx]),' ', str(via_map[1,via_idx])])
                f.write('\n')
        
"""
Write selected tree info (if it is different from the original tree) into a file for CUGR to read
write_tree: whetehr or not to write the tree info (for first round, no need to do that)

tree_p_index2pattern_index: tree_p_index2pattern_index[i] means the tree index of the i-th candidate
Return 
"""
def write_tree_result(nets,tree_p, tree_p_index_full, tree_p_index2pattern_index,data_name,write_tree):
    max_p, selected_index = scatter_max(tree_p,tree_p_index_full)
    selected_tree_full_index =  torch.zeros_like(tree_p_index_full, dtype = torch.bool)
    selected_tree_full_index[selected_index] = True # selected_tree_full_index[i] means whether the i-th tree is selected
    selected_trees = tree_p_index2pattern_index[selected_index.cpu()]
    non_orig_tree_num = 0
    if write_tree:
        with open('./CUGR2_guide/CUgr_'+data_name+'_tree.txt','w') as f:
            for idx in selected_trees:
                if idx[1] != 0:
                    non_orig_tree_num += 1
                    f.writelines([nets[idx[0]].net_name, ' ', str(nets[idx[0]].net_ID), ' ', str(len(nets[idx[0]].pins[idx[1]]))])
                    f.write('\n')
                    # write the tree (nets[idx[0]].pins[idx[1]])
                    for pinIdx, pin in enumerate(nets[idx[0]].pins[idx[1]]):
                        f.writelines([str(pinIdx), ' ', str(pin.x),' ', str(pin.y), ' ', str(pin.physical_pin_layers[0]), ' ', str(pin.physical_pin_layers[1])])
                        # write child in pin.child_pins
                        if pin.child_pins is not None:
                            for child in pin.child_pins:
                                f.writelines([' ', str(child)])
                        f.write('\n')
    print(" # of selected trees: ", len(selected_trees), " # of NON-orig trees: ", non_orig_tree_num)
    return selected_tree_full_index
"""
Return y_step_size, y_step_count, x_step_size, x_step_count 
based on dataname
"""
def get_ispd_size(data_name):
    # READ def file for the shape info,
    # def file is stored in `/scratch/weili3/cu-gr-2/benchmark/$data_name/$data_name.input.def`
    # the shape info is stored in the lines starting with `GCELLGRID`
    y_step_size = [0,0]
    y_step_count = [0,0]
    x_step_size = [0,0]
    x_step_count = [0,0]

    with open("/scratch/weili3/cu-gr-2/benchmark/{}/{}.input.def".format(data_name,data_name),'r') as f:
        lines = f.readlines()
        gcell_line = 0
        for line in lines:
            if line.startswith("GCELLGRID"):
                line = line.split() # The line is like GCELLGRID Y 288000 DO 2 STEP 4000 ;
                if line[1] == "X": # calculate x_step_size
                    is_final_step = int(int(line[2]) > 0)
                    x_step_size[is_final_step] = int(line[6])
                    x_step_count[is_final_step] = int(line[4]) - 1
                elif line[1] == "Y": # calculate y_step_size
                    is_final_step = int(int(line[2]) > 0)
                    y_step_size[is_final_step] = int(line[6])
                    y_step_count[is_final_step] = int(line[4]) - 1
                gcell_line += 1
            
            if gcell_line >= 4:
                break

    print(data_name, " shape info: ", y_step_size, y_step_count, x_step_size, x_step_count)
    return y_step_size, y_step_count, x_step_size, x_step_count


"""
For each 2-pin net, if the picked candidate (argmax) goes through some overflows, 
then we enumerate C/Z patterns to find the top-2 (top 5 for pattern_level = 3) candidate with the least objective.
Require:
    hor_congestion Tensor(xmax * ymax -1), demand - capacity, if >0, the congested
    ver_congestion Tensor(xmax -1 * ymax), demand - capacity, if >0, the congested
    pattern_level: 2: add C only, 3: add C and Z
    others are just for updating the candidate pool
"""
@torch.no_grad()
def add_CZ(p,p_index_full,p_index2pattern_index,hor_path,ver_path,wire_length_count, via_info, tree_index_per_candidate, 
           candidate_pool, hor_congestion, ver_congestion, pattern_level,args):
    assert pattern_level == 2 or pattern_level == 3, "pattern_level should be 2 or 3"
    xmax = args.xmax
    ymax = args.ymax
    device = args.device
    _, argmax = scatter_max(p,p_index_full)
    hor_is_cong = torch.matmul((hor_congestion > 0).to(torch.float32), hor_path).bool() 
    ver_is_cong = torch.matmul((ver_congestion > 0).to(torch.float32), ver_path).bool() 
    is_cong = (hor_is_cong + ver_is_cong).nonzero(as_tuple=True)[0] # the index of congested 2-pin nets
    if is_cong.numel() == 0:
        print("No congested net, return")
        return None
    # argmax is_cong result, intersection
    argmax_is_cong =np.intersect1d(argmax.cpu().numpy(), is_cong.cpu().numpy())
    all_via_map,via_count = via_info
    new_p_index_full = [] # p_index_full
    new_p_index2pattern_index = [] # p_index2pattern_index
    new_hor_path = []
    new_ver_path = []
    new_wire_length_count = []
    new_via_map = []
    new_via_count = []
    new_tree_index_per_candidate = []
    new_p = []
    print("%d/%d congested 2-pin nets are congested" % (len(argmax_is_cong), len(argmax)))
    # get 2-pin net candidate index     
    for idx in argmax_is_cong:
        # first, delete the current candidate (removing path and via influence)
        this_hor_path = hor_path.t()[idx].to_dense() # xmax * ymax -1
        this_ver_path = ver_path.t()[idx].to_dense()
        this_via_map = all_via_map.t()[idx].to_dense().view(xmax,ymax) # xmax * ymax
        this_hor_demmand = this_hor_path + ((this_via_map[:,:-1] + this_via_map[:,1:]) * args.via_layer / 2).flatten()
        this_ver_demmand = this_ver_path + ((this_via_map[:-1,:] + this_via_map[1:,:]) * args.via_layer / 2).flatten()
        this_hor_congestion = (hor_congestion - this_hor_demmand).float() # xmax * ymax -1
        this_ver_congestion = (ver_congestion - this_ver_demmand).float()
        # and then enumerate all C/Z patterns (corresponding edge_path and via info)
        this_idx = p_index2pattern_index[idx]
        this_pin = candidate_pool[this_idx[0]][this_idx[1]][this_idx[2]][0][0] 
        this_parent_pin = candidate_pool[this_idx[0]][this_idx[1]][this_idx[2]][0][1]
        result = []
        if pattern_level >= 2:
            # skip z pattern if the either this_pin.x = this_parent_pin.x or this_pin.y = this_parent_pin.y
            if this_pin.x != this_parent_pin.x and this_pin.y != this_parent_pin.y:
                result += enumerate_z(this_pin, this_parent_pin,args)
        if pattern_level >= 3:
            result += enumerate_c(this_pin, this_parent_pin,args)
        if len(result) == 0:
            continue
        wl_list = []
        via_map_list = []
        hor_list = []
        ver_list = []
        # print("before enumerate results ", torch.cuda.memory_allocated() / 1024 / 1024, "MB")
        for _, candidate in enumerate(result): # two_pin_net[0] is the tuple of (pin, parent_pin)
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
        # print("after enumerate results ", torch.cuda.memory_allocated() / 1024 / 1024, "MB")
        this_wire_length_count = torch.tensor(wl_list).to(device)
        this_via_map = torch.stack(via_map_list,dim = 1).to(device).float()
        this_via_count = this_via_map.sum(dim = 0).to_dense() # (n)
        this_hor_path = torch.stack(hor_list,dim = 1).to(device) # (xmax * (ymax - 1), n)
        this_ver_path = torch.stack(ver_list,dim = 1).to(device)

        # print("stack enumerate results ", torch.cuda.memory_allocated() / 1024 / 1024, "MB")
        dense_via_map = this_via_map.to_dense().view(xmax,ymax,-1) # (xmax, ymax, n)
        # print("dense_via_map", torch.cuda.memory_allocated() / 1024 / 1024, "MB")
        this_hor_via = ((dense_via_map[:,:-1,:] + dense_via_map[:,1:,:]) * args.via_layer / 2).view(xmax*(ymax-1), -1) # (xmax * (ymax - 1),n)
        this_ver_via = ((dense_via_map[:-1,:,:] + dense_via_map[1:,:,:]) * args.via_layer / 2).view((xmax-1)*ymax, -1) # ((xmax - 1) * ymax,n)
        # print("this_hor_via ", torch.cuda.memory_allocated() / 1024 / 1024, "MB")
        this_hor_demand = this_hor_via + this_hor_path
        this_ver_demand = this_ver_via + this_ver_path
        # print("this_hor_demand ", torch.cuda.memory_allocated() / 1024 / 1024, "MB")
        this_hor_cong_repeat = this_hor_congestion.view(-1,1).expand_as(this_hor_demand).clone() # (xmax * (ymax - 1), n)
        this_ver_cong_repeat = this_ver_congestion.view(-1,1).expand_as(this_ver_demand).clone() # ((xmax - 1) * ymax, n)
        # print("this_ver_cong_repeat ", torch.cuda.memory_allocated() / 1024 / 1024, "MB")
        all_hor_congestion = this_hor_cong_repeat.add(this_hor_demand) # (xmax * (ymax - 1), n)
        all_ver_congestion = this_ver_cong_repeat.add(this_ver_demand) # ((xmax - 1) * ymax, n)
        
        # print("before calculate cost ", torch.cuda.memory_allocated() / 1024 / 1024, "MB")

        # and then calculate the new overflow, pick the top-k candidates as new candidates
        overflow_cost = torch.sum(torch.nn.CELU(alpha = 2.0)(all_ver_congestion), dim = 0) + torch.sum(torch.nn.CELU(alpha = 2.0)(all_hor_congestion),dim=0)
        total_cost = overflow_cost + this_wire_length_count*args.wl_coeff + this_via_count*args.via_coeff
        # pick the top-k candidates as new candidates
        num_candidate = 1
        if pattern_level == 3:
            num_candidate = 2
        num_candidate = min(num_candidate, total_cost.shape[0])
        topk_cost, topk_index = torch.topk(total_cost, k = num_candidate, largest = False)
        # update candidate pool 
        for selected_idx in topk_index:
            candidate_pool[this_idx[0]][this_idx[1]][this_idx[2]][1].append(result[selected_idx])

        # print("before append new candidates ", torch.cuda.memory_allocated() / 1024 / 1024, "MB")
        # add the new candidates to the candidate pool
        # add p_index_full
        this_p_index = p_index_full[idx]
        new_p_index_full.append(torch.ones(num_candidate, dtype=torch.int64, device=device) * this_p_index)
        # add p_index2pattern_index
        this_p_index2pattern_index = p_index2pattern_index[idx]
        this_new_p_index2pattern_index = np.zeros((num_candidate,4), dtype=np.int64)
        this_new_p_index2pattern_index[:,0] = this_p_index2pattern_index[0]
        this_new_p_index2pattern_index[:,1] = this_p_index2pattern_index[1]
        this_new_p_index2pattern_index[:,2] = this_p_index2pattern_index[2]
        # this_new_p_index2pattern_index[:,3] = arrange(p_index2pattern_index[3]+1, p_index2pattern_index[3] + num_candidate+1)
        this_new_p_index2pattern_index[:,3] = np.arange(this_p_index2pattern_index[3]+1, this_p_index2pattern_index[3] + num_candidate+1)
        new_p_index2pattern_index.append(this_new_p_index2pattern_index)

        # add new_hor_path
        this_new_hor_path = this_hor_path.index_select(1,topk_index)
        new_hor_path.append(this_new_hor_path)
        # add new_ver_path
        this_new_ver_path = this_ver_path.index_select(1,topk_index)
        new_ver_path.append(this_new_ver_path)

        # add new wire length count
        this_new_wire_length_count = this_wire_length_count[topk_index]
        new_wire_length_count.append(this_new_wire_length_count)
        # add new via count
        this_new_via_count = this_via_count[topk_index]
        new_via_count.append(this_new_via_count)
        # add new via map
        this_new_via_map = this_via_map.index_select(1,topk_index)
        new_via_map.append(this_new_via_map)
        # add new tree_index_per_candidate
        this_tree_index_per_candidate = tree_index_per_candidate[idx]
        new_tree_index_per_candidate += [this_tree_index_per_candidate] * num_candidate
        # add new p values
        this_p = p[idx]
        min_cost = torch.min(topk_cost)
        max_cost = torch.max(topk_cost)
        if max_cost - min_cost < 1e-6: # for nu
            topk_p = torch.ones_like(topk_cost) * this_p
        else:
            topk_p =  this_p - ((topk_cost - min_cost) / (max_cost - min_cost) * 2) # min_cost p = this_p, max_cost p = this_p - 2, then, their probablity after softmax will be e^2:1
        if args.use_gumble:
            gumbels = (-torch.empty_like(topk_p, memory_format=torch.legacy_contiguous_format).exponential_().log())
            topk_p = topk_p + gumbels
        new_p.append(topk_p)
        # print("final memory usage: ", torch.cuda.memory_allocated() / 1024 / 1024, "MB")

    # append these values to the candidate pool
    # append p_index_full
    new_p_index_full = torch.cat(new_p_index_full, dim = 0)
    p_index_full = torch.cat([p_index_full, new_p_index_full], dim = 0)
    # p_index2pattern_index
    new_p_index2pattern_index = np.concatenate(new_p_index2pattern_index, axis = 0)
    p_index2pattern_index = np.concatenate([p_index2pattern_index, new_p_index2pattern_index], axis = 0)
    # append new_hor_path
    new_hor_path = torch.cat(new_hor_path, dim = 1)
    hor_path = torch.cat([hor_path, new_hor_path], dim = 1).float()
    # append new_ver_path
    new_ver_path = torch.cat(new_ver_path, dim = 1)
    ver_path = torch.cat([ver_path, new_ver_path], dim = 1)
    # append new wire length count
    new_wire_length_count = torch.cat(new_wire_length_count, dim = 0)
    wire_length_count = torch.cat([wire_length_count, new_wire_length_count], dim = 0)
    # append new via count
    new_via_count = torch.cat(new_via_count, dim = 0)
    via_count = torch.cat([via_count, new_via_count], dim = 0)
    # append new via map
    new_via_map = torch.cat(new_via_map, dim = 1)
    all_via_map = torch.cat([all_via_map, new_via_map], dim = 1)
    # append new tree_index_per_candidate
    new_tree_index_per_candidate = torch.tensor(new_tree_index_per_candidate, dtype=torch.int64, device=device)
    tree_index_per_candidate = torch.cat([tree_index_per_candidate, new_tree_index_per_candidate], dim = 0)
    # append new p
    new_p = torch.cat(new_p, dim = 0)
    
    p = torch.cat([p, new_p], dim = 0)

    return p_index_full, p_index2pattern_index, hor_path, ver_path, wire_length_count, via_count, all_via_map, tree_index_per_candidate, p, candidate_pool






"""
Enumerate all Z patterns for a congested 2-pin net
"""
def enumerate_z(pin, parent_pin,args):
    z_step = 1
    x_width = abs(pin.x - parent_pin.x)
    y_width = abs(pin.y - parent_pin.y)
    num_z = int((x_width - 1 + y_width - 1) / z_step)
    if num_z > args.max_z:
        z_step = math.ceil((x_width - 1 + y_width - 1) / args.max_z)
        num_z = int((x_width - 1 + y_width - 1) / z_step)

    if num_z <= 0:
        return []
    result = []
    random_offset = random.randint(1, z_step)
    for z_index in range(num_z):
        turning_x = min(pin.x, parent_pin.x) + ((z_index) * z_step + random_offset)
        if turning_x >= max(pin.x, parent_pin.x):
            turning_x = min(pin.x, parent_pin.x)
            turning_y = min(pin.y, parent_pin.y) + ((z_index) * z_step + random_offset) - (
                        max(pin.x, parent_pin.x) - min(pin.x, parent_pin.x) - 1)
            assert turning_y < max(pin.y, parent_pin.y)
        else:
            turning_y = min(pin.y, parent_pin.y)
        single_z_result = z_pattern_routing(pin, parent_pin, (turning_x, turning_y), args.xmax, args.ymax)
        result.append(single_z_result)
    return result


"""
Enumerate all C patterns for a congested 2-pin net
"""
def enumerate_c(pin, parent_pin, args):
    c_step = 5
    max_c_out_ratio = args.max_c_out_ratio
    x_width = abs(pin.x - parent_pin.x)
    y_width = abs(pin.y - parent_pin.y)
    xmax = args.xmax
    ymax = args.ymax
    # the maximum extend width along x axis (negative direction)
    extend_x_min = min(x_width * max_c_out_ratio, min(pin.x, parent_pin.x)) 
    # the maximum extend width along x axis (positive direction)
    extend_x_max = min(x_width * max_c_out_ratio, xmax - max(pin.x, parent_pin.x) - 1)
    # the maximum extend width along y axis (negative direction)
    extend_y_min = min(y_width * max_c_out_ratio, min(pin.y, parent_pin.y))
    # the maximum extend width along y axis (positive direction)
    extend_y_max = min(y_width * max_c_out_ratio, ymax - max(pin.y, parent_pin.y) - 1)
    total_extend_length = extend_x_min + extend_x_max + extend_y_min + extend_y_max
    num_c = int(total_extend_length / c_step)
    if num_c > args.max_c:
        c_step = math.ceil(total_extend_length / args.max_c)
        num_c = int(total_extend_length / c_step)
    result = []
    num_c = int(total_extend_length / c_step)
    # random from 1 to c_step
    random_offset = random.randint(1, c_step)
    if num_c > 0:
        for c_index in range(num_c):
            # start from opt1 ((c_index) * c_step + random_offset) <= extend_y_min
            if ((c_index) * c_step + random_offset) <= extend_y_min:
                turning_x = min(pin.x, parent_pin.x)
                turning_y = min(pin.y, parent_pin.y) - ((c_index) * c_step + random_offset)
                assert turning_y >= 0
            # opt2 extend_y_min < ((c_index) * c_step + random_offset) <= extend_y_min + extend_y_max
            elif ((c_index) * c_step + random_offset) <= extend_y_min + extend_y_max:
                turning_x = min(pin.x, parent_pin.x)
                turning_y = max(pin.y, parent_pin.y) + ((c_index) * c_step + random_offset) - extend_y_min
                assert turning_y < ymax
            # opt3 extend_y_min + extend_x_min < ((c_index) * c_step + random_offset) <= extend_y_min + extend_x_min + extend_y_max
            elif ((c_index) * c_step + random_offset) <= extend_y_min + extend_x_min + extend_y_max:
                turning_x = min(pin.x, parent_pin.x) - ((c_index) * c_step + random_offset) + extend_y_min + extend_y_max
                turning_y = min(pin.y, parent_pin.y)
                assert turning_x >= 0
            # opt4 extend_y_min + extend_x_min + extend_y_max < ((c_index) * c_step + random_offset) <= extend_y_min + extend_x_min + extend_y_max + extend_x_max
            elif ((c_index) * c_step + random_offset) <= extend_y_min + extend_x_min + extend_y_max + extend_x_max:
                turning_x = max(pin.x, parent_pin.x) + ((c_index) * c_step + random_offset) - extend_y_min - extend_x_min - extend_y_max
                turning_y = min(pin.y, parent_pin.y)
                assert turning_x < xmax
            else:
                assert False
            single_c_result = z_pattern_routing(pin, parent_pin, (turning_x, turning_y), xmax, ymax)
            result.append(single_c_result)
    return result
        


"""
Require:
    RouteNets: a list of Net objects
    cugr2_dir_path: the path to the CUGR2, it should includes the following four files:
        1. CUGR2_tree.txt, the new tree topology after phase 2
        2. CUGR2_maze.txt, the maze routing result after phase 3 (which means routing nets is in congested area)
        3. dgr_tree.txt, the new tree topology using our L-shape after phase 2
        4. dgr_maze.txt, the maze routing result using our L-shape after phase 3
        Currently, we do not use maze.txt, in our design maze.txt is only used to identify the congested nets. Congested nets will have Z/C shape candidates
        But now, we tend to design an iterative flow, which foucs on congested 2-pin net, rather than the whole net
"""
def read_new_tree(RouteNets,cugr2_dir_path):
    NetID2NetIdx = {} # map netID to netIdx of RouteNets
    for netIdx, net in enumerate(RouteNets):
        NetID2NetIdx[net.net_ID] = netIdx
    CUGR2_p2 = read_tree(os.path.join(cugr2_dir_path,'CUGR2_tree.txt'))
    for net in CUGR2_p2:
        netIdx = NetID2NetIdx[net.net_ID]
        RouteNets[netIdx].pins += net.pins # pins: TreeIdx -> PinIdx
    DGR_p2 = read_tree(os.path.join(cugr2_dir_path,'dgr_tree.txt'))
    both_congested = 0 # number of nets that are congested in both CUGR2 and DGR
    for net in DGR_p2:
        netIdx = NetID2NetIdx[net.net_ID]
        if len(RouteNets[netIdx].pins) > 1:
            both_congested += 1
        RouteNets[netIdx].pins += net.pins
    print("New Tree Read Finished. CUGR2 has {} congested nets, DGR has {} congested nets, {} congested in both".format(len(CUGR2_p2),len(DGR_p2),both_congested))
    return RouteNets


""" 
Given a tree path, read the tree and return the list of nets
"""
def read_tree(tree_path):
    result = []
    with open(tree_path, 'r') as f:
        for line in f.readlines():
            line = line.split()
            stored = False
            if len(line) == 0:
                # store the net into Net object
                if not stored:
                    # total num pins is the number of True in visited
                    total_num_pins = sum(visited)
                    this_net = Net(net_name, net_indx,pins = [pin_list[:total_num_pins]],num_pins = num_pins)
                    stored = True
                    result.append(this_net)
                continue
            if line[0] == 'tree':
                stored = False
                net_name = line[1]
                num_pins = int(line[2])
                net_indx= int(line[3])
                # here, we set the pin_list to be a list of 200 empty pins as placeholder
                pin_list = [Pin(0,0) for i in range(10*num_pins)]
                visited = [False for i in range(10*num_pins)]
                gcell_list = []
            else:
                x = int(line[0])
                y = int(line[1])
                low_layer = int(line[2])
                high_layer = int(line[3])
                pin_index = int(line[4])
                parent_indx = int(line[5])
                physical_pin_layers = [low_layer,high_layer]
                is_steiner = False if low_layer <= high_layer else True
                assert not visited[pin_index], 'pin_index %d is visited twice' % pin_index
                # create a new pin, but keep the parent pin (parent pin might be already set) and child_pins
                this_pin = Pin(x,y,is_steiner = is_steiner,parent_pin = pin_list[pin_index].parent_pin, child_pins = pin_list[pin_index].child_pins, physical_pin_layers = physical_pin_layers)
                pin_list[pin_index] = this_pin
                visited[pin_index] = True
                pin_list[pin_index].set_parent(parent_indx)
                if parent_indx >= 0:
                    pin_list[parent_indx].add_child(pin_index)
    return result

# print data read into the args
def print_data_stat(args):
    print("data_name: ", args.data_name)
    print("xmax: ", args.xmax)
    print("ymax: ", args.ymax)
    print("num_nets: ", args.num_nets)
    print("num_layers: ", args.num_layers)
