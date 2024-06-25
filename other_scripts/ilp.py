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
Scripts for ILP model and cvxpy solver
"""

import numpy as np
import cvxpy as cp
import math

"""
translate the data structure to data structure used by solve_by_lp()
    Require:
        RoutingRegion: routing_region object
        hor_path: sparse tensor, (xmax * ymax - 1, n), n is the number of candidates
        ver_path: sparse tensor, (xmax - 1 * ymax, n), n is the number of candidates
    Return:
        hor_cap: (xmax * ymax - 1) horizontal capacity matrix
        ver_cap: (xmax - 1 * ymax) vertical capacity matrix
        hor_path: SciPy sparse , (xmax * ymax - 1, n), n is the number of candidates
        ver_path: SciPy sparse, (xmax - 1 * ymax, n), n is the number of candidates
"""
def LP_data_prep(routing_region, hor_path, ver_path):
    hor_cap = routing_region.cap_mat[0].cpu().numpy().flatten()
    ver_cap = routing_region.cap_mat[1].cpu().numpy().flatten()
    hor_path = hor_path.cpu().coalesce()
    from scipy.sparse import coo_matrix
    # hor_path has shape (n, xmax, ymax - 1), n is the number of candidates, we need to reshape it to (xmax * ymax - 1, n)
    idx = hor_path.indices().numpy()
    values = hor_path.values().numpy()
    hor_path = coo_matrix((values, idx), shape=(hor_path.shape))
    ver_path = ver_path.cpu().coalesce()
    # ver_path has shape (n, xmax - 1, ymax), n is the number of candidates, we need to reshape it to (xmax - 1 * ymax, n)
    idx = ver_path.indices().numpy()
    values = ver_path.values().numpy()
    ver_path = coo_matrix((values, idx), shape=(ver_path.shape))
    return hor_cap, ver_cap, hor_path, ver_path

"""
solve the routing problem by ILP directly
    Require:
        hor_cap: (xmax * ymax - 1) horizontal capacity matrix
        ver_cap: (xmax - 1 * ymax) vertical capacity matrix
        hor_path: SciPy sparse , (xmax * ymax - 1, n), n is the number of candidates
        ver_path: SciPy sparse, (xmax - 1 * ymax, n), n is the number of candidates
        p_index: List(int), 
            given the probabilities p, p[pattern_index[i]:pattern_index[i+1]] is the probability distribution for the i_th 2-pin net with multiple candidates (one candidate does not require probability distribution)
            p_index[-1] is the length of the candidate_pool with multiple candidates
        is_ilp: bool, if True, solve the ILP problem, otherwise, solve the LP problem (p is continuous)
        pow: bool, if True, use the power-based model, otherwise, use the relu 06/19: Deprecated, it is not convex by cvxpy (it is if x > 0 actually)
    Return:
        p: (n) probability distribution for each candidate
        obj: objective value
"""
def solve_by_lp(hor_cap, ver_cap, hor_path, ver_path,p_index, is_ilp = True,pow = False):
    n = hor_path.shape[1]
    constraints = []
    if is_ilp:
        p = cp.Variable(n, boolean=True)
    else:
        p = cp.Variable(n)
        # add constraints p >= 0
        constraints = [p >= 0]
        constraints.append(p <= 1)
    for i in range(len(p_index)-1):
        constraints.append(cp.sum(p[p_index[i]:p_index[i+1]]) == 1)
    # add constraints p <= 1
    # objective is min(pos(hor_path @ p - hor_cap) + pos(ver_path @ p - ver_cap))
    # if pow:
    #     objective = cp.Minimize(cp.sum(1/ (1+cp.exp(hor_cap - (hor_path @ p)))) + cp.sum(1/ (1+cp.exp(ver_cap - (ver_path @ p)))))
    # else:
    objective = cp.Minimize(cp.sum(cp.pos(hor_path @ p - hor_cap)) + cp.sum(cp.pos(ver_path @ p - ver_cap)))
    prob = cp.Problem(objective, constraints)
    prob.solve()
    return p.value, prob.value

