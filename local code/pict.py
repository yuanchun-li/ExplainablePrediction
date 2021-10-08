# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 13:30:43 2021
This program is used to count the number of occurrences of the recitation code and make a graph.

"""

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

from collections import Counter

count_arr = []
cnt = 1
limit = 1000000

with open('C:/Users/v-weixyan/Desktop/CodeGPT-Py150/py150_test/test_recitation.json', 'r', encoding="utf-8") as f:
    for jsonstr in f.readlines():
        item = json.loads(jsonstr)
        count_arr.append(item["count"])
        cnt += 1
        if cnt == limit:
            break

count_arr.sort()
count_dict = dict(Counter(count_arr))
a = list(count_dict.keys())
b = list(count_dict.values())

df = pd.DataFrame({"x-axis": a,"y-axis": b})

sns.barplot("y-axis", "x-axis", orient='h', palette="Accent_r",data=df)
#plt.xticks(rotation=90)
#plt.yticks(rotation=90)
plt.show()