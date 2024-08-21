import os
import pandas as pd
from scipy.stats import entropy

# 假設敏感屬性為'salary-class'
sensitive_attr = 'salary-class'

data = pd.read_csv('data/adult.csv', sep=';')
k = 30

qis = ['sex', 'age', 'race', 'marital-status', 'education', 'native-country', 'workclass', 'occupation']


def summarized(partition, dim):
    for qi in qis:
        partition = partition.sort_values(by=qi)
        if partition[qi].iloc[0] != partition[qi].iloc[-1]:
            s = f"[{partition[qi].iloc[0]}-{partition[qi].iloc[-1]}]"
            if qi == 'sex' and (s == '[Female-Male]' or s == '[Male-Female]'):
                s = '*'
            partition[qi] = [s] * partition[qi].size

    return partition


def anonymize(partition, ranks):
    dim = ranks[0][0]

    partition = partition.sort_values(by=dim)
    si = partition[dim].count()
    mid = si // 2

    lhs = partition[:mid]
    rhs = partition[mid:]

    if len(lhs) >= k and len(rhs) >= k:
        return pd.concat([anonymize(lhs, ranks), anonymize(rhs, ranks)])

    return summarized(partition, dim)


def mondrian(partition):
    ranks = {}

    for qi in qis:
        # 計算每個qis內的種類
        ranks[qi] = len(set(partition[qi]))

    # sort ranks
    ranks = sorted(ranks.items(), key=lambda t: t[1], reverse=True)
    # print(ranks)

    return anonymize(partition, ranks)


result = mondrian(data)

output_dir = 'results'
os.makedirs(output_dir, exist_ok=True)

# 將結果保存到CSV文件
output_file = os.path.join(output_dir, f'anon_k={k}.csv')
result.to_csv(output_file, index=False)