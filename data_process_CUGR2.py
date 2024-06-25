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
Given the output generated from the CUGR2, we need to process the data to fit into our algorithm
Saved data results:
    results[i]['net'][j] is the j_th Net object in i benchmark; results[i]['region'] = routing region
"""
import data
import os
import torch
import sys

cugr2_dir = sys.argv[1] 
benchmark_path = sys.argv[2] 
data_names = [d for d in os.listdir(benchmark_path) if os.path.isdir(os.path.join(benchmark_path, d))]
print("data_names: ", data_names)
results = {} # results[i]['net'][j] is the j_th Net object in i benchmark; results[i]['region'] = routing region
for data_name in data_names:
    # cd {cugr2_dir}/run/ 
    print("data_name: ", data_name)
    results[data_name] = {}
    os.chdir('{cugr2_dir}/run/'.format(cugr2_dir = cugr2_dir))
    os.system('rm {benchmark_path}/{data_name}/{data_name}.output2'.format(benchmark_path = benchmark_path, data_name = data_name))
    os.system('./route -lef {benchmark_path}/{data_name}/{data_name}.input.lef -def {benchmark_path}/{data_name}/{data_name}.input.def -output {benchmark_path}/{data_name}/{data_name}.output2 -threads 1 -sort 1'.format(benchmark_path = benchmark_path, data_name = data_name))
    # wait until the output file is generated
    while not os.path.exists('{benchmark_path}/{data_name}/{data_name}.output2'.format(benchmark_path = benchmark_path, data_name = data_name)):
        pass
    print('Processing data: ', data_name)
    # read 'layout.txt'

    with open('./layout.txt', 'r') as f:
        # Read the first line
        sizes = f.readline().strip().split()
        results[data_name]['xmax'] = int(sizes[0])
        results[data_name]['ymax'] = int(sizes[1])
        results[data_name]['m2_pitch'] = float(sizes[2])
        
        # Read the subsequent lines for layer details
        results[data_name]['layers'] = []
        for line in f:
            layer_details = line.strip().split()
            layer_info = {
                'layer_index': int(layer_details[0]),
                'pitch': float(layer_details[1]),
                'unit_length_short_cost': float(layer_details[2]),
                'layerMinLength': float(layer_details[3])
            }
            results[data_name]['layers'].append(layer_info)


    # read capacity3D
    first_layer = True
    hor_cap_list = []
    ver_cap_list = []
    xmax = results[data_name]['xmax']
    ymax = results[data_name]['ymax']
    # we load the fixed2D.txt and capacity2D.txt, which stores the blockage and capacity information for each gcell
    with open('capacity3D.txt','r') as f2:
        for line2 in f2.readlines():
            # if the line only contains one integer, it is the direction indicator
            if len(line2.split()) == 1:
                # if not first layer, then, we need to save the previous layer
                if not first_layer:
                    # save the previous layer
                    if direction == 1:
                        hor_cap_list.append(hor_cap)
                    else:
                        ver_cap_list.append(ver_cap)
                direction = int(line2.split()[0])
                # index is the row/col index of the gcell
                index = 0
                if first_layer:
                    first_layer = False
                    is_hor = (direction == 1)
                hor_cap = torch.zeros((xmax, ymax-1))
                ver_cap = torch.zeros((xmax-1, ymax))
                continue
            # else, it encodes the cap/fix information, each row is a row of gcells
            line2 = torch.tensor([float(x) for x in line2.split()])
            # direction = 0: horizontal, 1: vertical
            if direction == 1:
                hor_cap[index] = line2
            else:
                ver_cap[index] = line2
            index += 1
    if direction == 1:
        hor_cap_list.append(hor_cap)
    else:
        ver_cap_list.append(ver_cap)
    RoutingGrid3d = data.routing_region_3D(xmax, ymax, (hor_cap_list,ver_cap_list), is_hor)
    results[data_name]['region3D'] = RoutingGrid3d

    # then, create 2D region cap values
    hor_cap = torch.stack(hor_cap_list).sum(dim = 0)
    ver_cap = torch.stack(ver_cap_list).sum(dim = 0)
    RoutingGrid = data.routing_region(xmax,ymax,cap_mat=(hor_cap, ver_cap))
    results[data_name]['region'] = RoutingGrid
    # load the output file tree.txt, which contains all nets information
    # each net starts with a line: 'netname num_pins need_route'
    # then num_pins lines, each line is a pin_index, is_steiner, x, y, child index
    result = []
    with open('./tree.txt', 'r') as f:
        for line in f.readlines():
            line = line.split()
            stored = False
            if len(line) == 0:
                # store the net into Net object
                if not stored:
                    # total num pins is the number of True in visited
                    total_num_pins = sum(visited)
                    this_net = data.Net(net_name, net_indx,pins = [pin_list[:total_num_pins]],num_pins = num_pins)
                    stored = True
                    result.append(this_net)
                continue
            if line[0] == 'tree':
                stored = False
                net_name = line[1]
                num_pins = int(line[2])
                net_indx= int(line[3])
                # here, we set the pin_list to be a list of 200 empty pins as placeholder
                pin_list = [data.Pin(0,0) for i in range(2*num_pins)]
                visited = [False for i in range(2*num_pins)]
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
                this_pin = data.Pin(x,y,is_steiner = is_steiner,parent_pin = pin_list[pin_index].parent_pin, child_pins = pin_list[pin_index].child_pins, physical_pin_layers = physical_pin_layers)
                pin_list[pin_index] = this_pin
                visited[pin_index] = True
                pin_list[pin_index].set_parent(parent_indx)
                if parent_indx >= 0:
                    pin_list[parent_indx].add_child(pin_index)
    results[data_name]['net'] = result

    edge_length = []
    # load 'edge_length.txt'. edge_length[0] is edge length array for horizontal edges, edge_length[1] is for vertical edges
    # in edge_length.txt, first line is the list of horizontal edge lengths, second line is the list of vertical edge lengths
    with open('./edge_length.txt', 'r') as f:
        for line in f.readlines():
            line = line.split()
            edge_length.append(torch.tensor([float(x) for x in line]))
    results[data_name]['edge_length'] = edge_length
    # save results
    # pickle.dump(results[data_name], open(data_name + '.pkl', 'wb'))
    torch.save(results[data_name], data_name + '.pt')

    
