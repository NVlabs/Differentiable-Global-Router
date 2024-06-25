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

# run different parameters for result analysis
data="${1:-ispd18_test5_metal5}"
benchmark_path="/scratch/weili3/cu-gr-2/benchmark"
cugr2="/scratch/weili3/cu-gr-2"
this_path="/home/weili3/Differentiable-Global-Router"
echo "Processing New Tree... data: $data"
python3 main.py --data_path $cugr2/run/$data.pt --read_new_tree True
cd $cugr2/run/
./route -lef $benchmark_path/$data/$data.input.lef -def $benchmark_path/$data/$data.input.def -output $benchmark_path/$data/$data.output4 -dgr $this_path/CUGR2_guide/CUgr_$data\_1\_0.txt -tree $this_path/CUGR2_guide/CUgr_$data\_tree.txt > $cugr2/GR_log/$data\_NewT.log
cd $cugr2
./drcu -lef $benchmark_path/$data/$data.input.lef -def $benchmark_path/$data/$data.input.def -thread 8 -guide $benchmark_path/$data/$data.output4 --output $cugr2/DR_result/NewT_$data.txt --tat 2000000000 > $cugr2/DR_log/$data\_NewT.log
cd $this_path
