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
Given the output generated from the CUGR, we need to process the data to fit into our algorithm
Saved data results:
    results[i]['net'][j] is the j_th Net object in i benchmark; results[i]['region'] = routing region
"""
import data
import os
import torch
import pickle
# ispd18_test{%d} from 1 to 10
# data_names = ['ispd18_test%d_metal5' % i for i in [5,6]]
data_names = ['ispd19_test%d_metal5' % i for i in [8,9]]
# data_names = ['ispd18_test%d' % i for i in range(5,6)]
# # data_names += ['ispd19_test%d' % i for i in range(1,8)]
cugr_dir = '/home/scratch.rliang_hardware/wli1/cu-gr'
results = {} # results[i]['net'][j] is the j_th Net object in i benchmark; results[i]['region'] = routing region
for data_name in data_names:
    # cd {cugr_dir}/run/ 
    results[data_name] = {}
    os.chdir('{cugr_dir}/run/'.format(cugr_dir = cugr_dir))
    os.system('rm {cugr_dir}/benchmark/{data_name}/{data_name}.output'.format(cugr_dir = cugr_dir, data_name = data_name))
    # excute shell commands: {cugr_dir}/run/iccad19gr -lef {cugr_dir}/{data_name}/{data_name}.input.lef -def {cugr_dir}/{data_name}/{data_name}.input.def -output {cugr_dir}/{data_name}/{data_name}.output -threads 8
    os.system('./iccad19gr -lef {cugr_dir}/benchmark/{data_name}/{data_name}.input.lef -def {cugr_dir}/benchmark/{data_name}/{data_name}.input.def -output {cugr_dir}/benchmark/{data_name}/{data_name}.output -threads 8'.format(cugr_dir = cugr_dir, data_name = data_name))
    # wait until the output file is generated
    while not os.path.exists('{cugr_dir}/benchmark/{data_name}/{data_name}.output'.format(cugr_dir = cugr_dir, data_name = data_name)):
        pass
    print('Processing data: ', data_name)
    # first load the layout.txt
    # first line, 2 integers, xmax, ymax
    # for the following lines, each line is seperated by a space, line[0] is layer index,
    # line[1] indicates it is horizontal or vertical, == xmax: horizontal, == ymax: vertical
    # line[2] and the following is the capacity for each track (grid)
    with open('./layout.txt', 'r') as f:
        line = f.readline()
        xmax, ymax = line.split()
        xmax, ymax = int(xmax), int(ymax)
    
    first_layer = True
    hor_cap_list = []
    ver_cap_list = []
    # we load the fixed2D.txt and capacity2D.txt, which stores the blockage and capacity information for each gcell
    with open('./fixed3D.txt', 'r') as f:
        with open('capacity3D.txt','r') as f2:

            for line in f.readlines():
                line2 = f2.readline()
                # if the line only contains one integer, it is the direction indicator
                if len(line.split()) == 2:
                    # if not first layer, then, we need to save the previous layer
                    if not first_layer:
                        # save the previous layer
                        if direction == 0:
                            hor_cap_list.append(hor_cap-hor_fix)
                        else:
                            ver_cap_list.append(ver_cap-ver_fix)
                    direction = int(line.split()[1])
                    # index is the row/col index of the gcell
                    index = 0
                    if first_layer:
                        first_layer = False
                        is_hor = (direction == 0)
                    hor_cap = torch.zeros((xmax, ymax-1))
                    ver_cap = torch.zeros((xmax-1, ymax))
                    hor_fix = torch.zeros((xmax, ymax-1))
                    ver_fix = torch.zeros((xmax-1, ymax))
                    continue
                # else, it encodes the cap/fix information, each row is a row of gcells
                line = torch.tensor([int(x) for x in line.split()])
                line2 = torch.tensor([int(x) for x in line2.split()])
                # direction = 0: horizontal, 1: vertical
                if direction == 0:
                    hor_cap[index] = line2
                    hor_fix[index] = line
                else:
                    ver_cap[:,index] = line2
                    ver_fix[:,index] = line
                index += 1
    if direction == 0:
        hor_cap_list.append(hor_cap-hor_fix)
    else:
        ver_cap_list.append(ver_cap-ver_fix)
    RoutingGrid3d = data.routing_region_3D(xmax, ymax, (hor_cap_list,ver_cap_list), is_hor)
    results[data_name]['region3D'] = RoutingGrid3d
    # we load the fixed2D.txt and capacity2D.txt, which stores the blockage and capacity information for each gcell
    with open('./fixed2D.txt', 'r') as f:
        with open('capacity2D.txt','r') as f2:
            hor_cap = torch.zeros((xmax, ymax-1))
            ver_cap = torch.zeros((xmax-1, ymax))
            hor_fix = torch.zeros((xmax, ymax-1))
            ver_fix = torch.zeros((xmax-1, ymax))
            for line in f.readlines():
                line2 = f2.readline()
                # if the line only contains one integer, it is the direction indicator
                if len(line.split()) == 1:
                    direction = int(line)
                    # index is the row/col index of the gcell
                    index = 0
                    continue
                # else, it encodes the cap/fix information, each row is a row of gcells
                line = torch.tensor([int(x) for x in line.split()])
                line2 = torch.tensor([int(x) for x in line2.split()])
                # direction = 0: horizontal, 1: vertical
                if direction == 0:
                    hor_cap[index] = line2
                    hor_fix[index] = line
                else:
                    ver_cap[:,index] = line2
                    ver_fix[:,index] = line
                index += 1
    # update cap by subtracting the fixed capacity
    hor_cap -= hor_fix
    ver_cap -= ver_fix
    RoutingGrid = data.routing_region(xmax,ymax,cap_mat=(hor_cap, ver_cap))
    results[data_name]['region'] = RoutingGrid
    # load the output file tree.txt, which contains all nets information
    # each net starts with a line: 'tree netname num_pins need_route'
    # then num_pins lines, each line is a pin_index, is_steiner, x, y, child index
    result = []
    with open('./tree.txt', 'r') as f:
        for line in f.readlines():
            line = line.split()
            stored = False
            if len(line) == 0:
                # store the net into Net object
                if not stored:
                    if need_route is False:
                        this_net = data.Net(net_name, 0,need_route= False, gcell_list = gcell_list)
                    else:
                        # total num pins is the number of True in visited
                        total_num_pins = sum(visited)
                        this_net = data.Net(net_name, 0,pins = [pin_list[:total_num_pins]],num_pins = num_pins)
                        stored = True
                    result.append(this_net)
                continue
            if line[0] == 'tree':
                stored = False
                net_name = line[1]
                num_pins = int(line[2])
                need_route = True if line[3] == '1' else False
                # here, we set the pin_list to be a list of 200 empty pins as placeholder
                pin_list = [data.Pin(0,0) for i in range(2*num_pins)]
                visited = [False for i in range(2*num_pins)]
                gcell_list = []
            else:
                if need_route is False:
                    gcell_value = (int(line[0]), int(line[1]), int(line[2])) # x,y,layer
                    gcell_list.append(gcell_value)
                else:
                    pin_index = int(line[0])
                    is_steiner = int(line[1])
                    x = int(line[2])
                    y = int(line[3])
                    child_index = int(line[4])
                    num_physical_pin = int(line[5])
                    physical_pin_layers = []
                    for i in range(num_physical_pin):
                        physical_pin_layers.append(int(line[6+i]))
                    if not visited[pin_index]:
                        # create a new pin, but keep the parent pin (parent pin might be already set) and child_pins
                        this_pin = data.Pin(x,y,is_steiner = is_steiner,parent_pin = pin_list[pin_index].parent_pin, child_pins = pin_list[pin_index].child_pins, physical_pin_layers = physical_pin_layers)
                        pin_list[pin_index] = this_pin
                        visited[pin_index] = True
                    if not visited[child_index]:
                        # if not visit child_index yet, we update the parent pin
                        pin_list[child_index].set_parent(pin_index)
                        pin_list[pin_index].add_child(child_index)
    results[data_name]['net'] = result
    # save results
    # pickle.dump(results[data_name], open(data_name + '.pkl', 'wb'))
    torch.save(results[data_name], data_name + '.pt')

    
