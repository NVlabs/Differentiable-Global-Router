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
import pickle
# ispd18_test{%d} from 1 to 10
data_names = ['ispd18_test%d_metal5' % i for i in [5,8]]
data_names += ['ispd19_test%d_metal5' % i for i in [7]]
cugr_dir = '/home/scratch.rliang_hardware/wli1/cu-gr'
cugr2_dir = '/home/scratch.rliang_hardware/wli1/cu-gr-2'
results = {} # results[i]['net'][j] is the j_th Net object in i benchmark; results[i]['region'] = routing region
for data_name in data_names:
    # cd {cugr_dir}/run/ 
    results[data_name] = {}
    os.chdir('{cugr2_dir}/run/'.format(cugr2_dir = cugr2_dir))
    # rm '{cugr_dir}/benchmark/{data_name}/{data_name}.output2'.format(cugr_dir = cugr_dir, data_name = data_name)
    os.system('rm {cugr_dir}/benchmark/{data_name}/{data_name}.output2'.format(cugr_dir = cugr_dir, data_name = data_name))
    os.system('./route -lef {cugr_dir}/benchmark/{data_name}/{data_name}.input.lef -def {cugr_dir}/benchmark/{data_name}/{data_name}.input.def -output {cugr_dir}/benchmark/{data_name}/{data_name}.output2 -threads 1'.format(cugr_dir = cugr_dir, data_name = data_name))
    # wait until the output file is generated
    while not os.path.exists('{cugr_dir}/benchmark/{data_name}/{data_name}.output2'.format(cugr_dir = cugr_dir, data_name = data_name)):
        pass
    print('Processing data: ', data_name)

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
    # save results
    # pickle.dump(results[data_name], open(data_name + '.pkl', 'wb'))
    torch.save(results[data_name], data_name + '.pt')

    
