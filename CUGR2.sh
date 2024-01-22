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
data="${1:-ispd18_test1}"
benchmark_path=$2
cugr2=$3
this_path=$4
mkdir $cugr2/GR_log
mkdir $cugr2/DR_log
mkdir $cugr2/DR_result
echo "Processing CUGR2 First Round... data: $data"
cd $cugr2/run/
./route -lef $benchmark_path/$data/$data.input.lef -def $benchmark_path/$data/$data.input.def -output $benchmark_path/$data/$data.CUGR2.guide -sort 0 > $cugr2/GR_log/$data\_CUGR2.log
cd $cugr2
./drcu -lef $benchmark_path/$data/$data.input.lef -def $benchmark_path/$data/$data.input.def -thread 32 -guide $benchmark_path/$data/$data.CUGR2.guide --output $cugr2/DR_result/CUGR2_$data.txt --tat 2000000000 > $cugr2/DR_log/$data\_CUGR2.log
cd $this_path
sh run_innovus.sh $data CUGR2
