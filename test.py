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
import torch

is_hor_list = [True, True, False, False, True, True, False, False,False, True, True]

# Convert the list to a PyTorch tensor
is_hor_tensor = torch.tensor(is_hor_list)

# Calculate the count tensor
diff = torch.diff(is_hor_tensor, prepend=torch.tensor([0], dtype=torch.bool))
C_tensor = torch.cumsum(diff == 1, dim=0)[diff == 0]

print(C_tensor)

exit()
value = torch.nn.Parameter(torch.ones(50, 3).cuda(), requires_grad=True)
indices = torch.ones(2,50).cuda()
s = torch.sparse_coo_tensor(indices, value, (2, 4,3)).cuda()
s = s.coalesce()
# backprop
s.sum().backward()

# y = torch.rand(100, 5000).cuda()
# initialize a sparse tensor with rand values
y = torch.sparse_coo_tensor([[1,2],[3,4]], [2,4], (100, 5000)).cuda().float()
y2 = torch.sparse_coo_tensor([[1,2],[3,4]], [2,4], (100, 5000)).cuda().float()
a = torch.cat([y,y2], dim=1)
import timeit
start = timeit.default_timer()
bmm = torch.bmm(y.unsqueeze(2),b.unsqueeze(1)) # Method of @fmassa
# backprop
bmm.sum().backward()

print("bmm time: ", timeit.default_timer() - start)
start = timeit.default_timer()
einsum = torch.einsum('bi,bj->bij', (y,x))
einsum.sum().backward()
print("einsum time: ", timeit.default_timer() - start)
print("bmm shape: ", bmm.shape)

