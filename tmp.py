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
A temporary file for data processing
"""
import torch
results = torch.load("/home/scratch.rliang_hardware/wli1/cu-gr/run/iccad19_benchmark.pt")

# mkdir data if not exist
import os
if not os.path.exists('data'):
    os.mkdir('data')
# for key in results, save results[key]
for key in results.keys():
    results[key]['region'].cap_mat = (-results[key]['region'].cap_mat[0], -results[key]['region'].cap_mat[1])
    torch.save(results[key], 'data/%s.pt' % key)
