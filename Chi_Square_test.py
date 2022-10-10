import csv

import pandas as pd
from scipy.stats import chi2_contingency

import seaborn as sns
import matplotlib.pyplot as plt

tmp_lst = []
with open('bc.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        tmp_lst.append(row)

df = pd.DataFrame(tmp_lst[1:], columns=tmp_lst[0])

chi_square_dict = {}
for column1 in df.keys():
    if column1 == "BC":
        continue

    chi_square_dict[column1] = []

    for column2 in df.keys():
        if column2 == "BC":
            continue

        cross_table = pd.crosstab(df[column1], df[column2], margins=True)
        chi_value, p_value, degree_of_freedom, expected_freq = chi2_contingency(cross_table)
        chi_square_dict[column1].append(p_value)

chi_square_table = pd.DataFrame(chi_square_dict, index=chi_square_dict.keys())

sns.heatmap(chi_square_table, annot=True)
plt.tick_params(axis='x', labelsize=7.5)
plt.xticks(rotation=80, fontweight='bold')
plt.show()





