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

# Assuming you have the DataFrame 'df' with columns 'via_coeff', 'pin_via_cost', and 'wl_coeff'
# For example, you can create a sample DataFrame as follows:
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns

# read step1.csv
df = pd.read_csv('step1.csv')
df2 = pd.read_csv('step2.csv')
# remove 'via_coeff' column from df2
df2 = df2.drop(columns=['via_coeff'])
# merge df and df2 per row
df = pd.concat([df, df2], axis=1)
# plot df, x axis is column 'via_coeff' in log scale, y axis is pin_via_cost column, and plot three lines, each line is a wl_coeff value
# Get the unique 'wl_coeff' values in the DataFrame
unique_wl_coeffs = df['wl_coeff'].unique()

# Create a log-scale plot with lines for each 'wl_coeff' value
plt.figure(figsize=(10, 6))
for coeff in unique_wl_coeffs:
    df_subset = df[df['wl_coeff'] == coeff]
    plt.plot(df_subset['via_coeff'], df_subset['pin_via_cost'], label=f'wl_coeff = {coeff}')
plt.xscale('log')  # Set the x-axis to log scale
plt.xlabel('via_coeff (log scale)')
plt.ylabel('pin_via_cost')
plt.title('Plot of pin_via_cost vs. via_coeff with different wl_coeff values')
plt.legend()
plt.grid(True)
plt.savefig('step2.png')
