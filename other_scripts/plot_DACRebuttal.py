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


import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
sns.set()
# read step1.csv
df = pd.read_csv('./ray_results/ispd18_test8_metal5_lr.csv')
# create four scatters plots in a figure, x axis is `config/learning_rate` in log scale, y axis is score, wirelength, via_count, first_overflow, respectively
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
sns.scatterplot(ax=axes[0, 0], x="config/learning_rate", y="score", data=df)
sns.scatterplot(ax=axes[0, 1], x="config/learning_rate", y="wire_length", data=df)
sns.scatterplot(ax=axes[1, 0], x="config/learning_rate", y="via_count", data=df)
sns.scatterplot(ax=axes[1, 1], x="config/learning_rate", y="first_overflow", data=df)
# log scale in x 
axes[0, 0].set_xscale('log')
axes[0, 1].set_xscale('log')
axes[1, 0].set_xscale('log')
axes[1, 1].set_xscale('log')

plt.savefig('scatter.png')

# # create four boxplots in a figure, x axis is `config/optimizer`, y axis is score, wirelength, via_count, first_overflow, respectively
# fig, axes = plt.subplots(2, 2, figsize=(10, 10))
# sns.boxplot(ax=axes[0, 0], x="config/scheduler", y="score", data=df)
# sns.boxplot(ax=axes[0, 1], x="config/scheduler", y="wire_length", data=df)
# sns.boxplot(ax=axes[1, 0], x="config/scheduler", y="via_count", data=df)
# sns.boxplot(ax=axes[1, 1], x="config/scheduler", y="first_overflow", data=df)
# plt.savefig('boxplot.png')
