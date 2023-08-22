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
Read nets from target_net.txt, and plot the nets.
"""

import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# read the nets from target_net.txt
# target_net.txt is ordered like this:
# net_name
# (
# x1, y1, x2, y2, layer_name
# x1, y1, x2, y2, layer_name
# ...
# )
# net_name2
# (
# ...
# )
with open('./target_net.txt', 'r') as f:
    nets = []
    x_min = 1e9
    x_max = -1e9
    y_min = 1e9
    y_max = -1e9
    layer_min = 10
    layer_max = -1
    for line in f.readlines():
        if line[0] == '(':
            nets.append([])
        elif line[0] == ')':
            pass
        elif line[0] == 'n':
            pass
        else:
            this_line = line.split(' ')
            this_x_min = int(this_line[0])
            this_x_max = int(this_line[2])
            this_y_min = int(this_line[1])
            this_y_max = int(this_line[3])
            this_layer = int(this_line[4].split('Metal')[1])
            nets[-1].append([this_x_min, this_y_min, this_x_max, this_y_max, this_layer])
            x_min = min(x_min, this_x_min)
            x_max = max(x_max, this_x_max)
            y_min = min(y_min, this_y_min)
            y_max = max(y_max, this_y_max)
            layer_min = min(layer_min, this_layer)
            layer_max = max(layer_max, this_layer)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for netidx, net in enumerate(nets):
        for edge in net:
            xmin, ymin, xmax, ymax, layer = edge
            zmin = layer
            zmax = layer + 1
            vertices = [
                (xmin, ymin, zmin),
                (xmax, ymin, zmin),
                (xmax, ymax, zmin),
                (xmin, ymax, zmin),
                (xmin, ymin, zmax),
                (xmax, ymin, zmax),
                (xmax, ymax, zmax),
                (xmin, ymax, zmax),
            ]
            faces = [
                [vertices[0], vertices[1], vertices[2], vertices[3]],  # Bottom face
                [vertices[4], vertices[5], vertices[6], vertices[7]],  # Top face
                [vertices[0], vertices[1], vertices[5], vertices[4]],  # Side face 1
                [vertices[1], vertices[2], vertices[6], vertices[5]],  # Side face 2
                [vertices[2], vertices[3], vertices[7], vertices[6]],  # Side face 3
                [vertices[3], vertices[0], vertices[4], vertices[7]],  # Side face 4
            ]
            # facecolors change with netidx
            facecolor = sns.color_palette("hls", len(nets))[netidx]
            ax.add_collection3d(Poly3DCollection(faces, facecolors=facecolor))
    # Set axis limits
    ax.set_xlim([x_min-1000, x_max + 1000])
    ax.set_ylim([y_min-1000, y_max + 1000])
    ax.set_zlim([layer_min-1, layer_max + 1])

    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('layer')


    plt.savefig('test.png')

        
