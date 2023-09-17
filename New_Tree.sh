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
# run different parameters for result analysis
data="${1:-ispd18_test1}"
benchmark_path=$2
cugr2=$3
this_path=$4
method="NewTree"
# first round CUGR (with new sort), generating fine-tuned tree
cd $cugr2/run/
./route -lef $benchmark_path/$data/$data.input.lef -def $benchmark_path/$data/$data.input.def -output $benchmark_path/$data/$data.CUGR2_NewSort.guide -sort 1 > $cugr2/GR_log/$data\_CUGR2_NewSort.log

cd $this_path
# first round DGR and second round CUGR2, generating fine-tuned tree
python3 main_stochastic.py --data_path $cugr2/run/$data.pt --output_name FirstRound

cd $cugr2/run/
./route -lef $benchmark_path/$data/$data.input.lef -def $benchmark_path/$data/$data.input.def -output $benchmark_path/$data/$data.FirstRound.guide -dgr $this_path/CUGR2_guide/CUgr_$data\_FirstRound.txt > $cugr2/GR_log/$data\_FirstRound.log

# second round DGR and third round CUGR2, generating final result
cd $this_path
python3 main_stochastic.py --data_path $cugr2/run/$data.pt  --read_new_tree True --output_name $method

cd $cugr2/run/
./route -lef $benchmark_path/$data/$data.input.lef -def $benchmark_path/$data/$data.input.def -output $benchmark_path/$data/$data.$method.guide -dgr $this_path/CUGR2_guide/CUgr_$data\_$method.txt -tree $this_path/CUGR2_guide/CUgr_$data\_tree.txt > $cugr2/GR_log/$data\_$method.log

# Run Detailed Routing
cd $cugr2
./drcu -lef $benchmark_path/$data/$data.input.lef -def $benchmark_path/$data/$data.input.def -thread 32 -guide $benchmark_path/$data/$data.$method.guide --output $cugr2/DR_result/${method}_$data.txt --tat 2000000000 > $cugr2/DR_log/$data\_$method.log

cd $this_path
sh run_innovus.sh $data $method