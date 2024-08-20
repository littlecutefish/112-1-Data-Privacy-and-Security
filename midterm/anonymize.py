import pandas as pd
from scipy.stats import entropy

# 假設敏感屬性為'salary-class'
sensitive_attr = 'salary-class'
l = 1
t = 1

data = pd.read_csv('data/adult.csv', sep=';')
k = 2

qis = ['sex', 'age', 'race', 'marital-status', 'education', 'native-country', 'workclass', 'occupation']

def check_l_diversity(partition, l):
    # 計算敏感屬性的多樣性
    diversity = partition[sensitive_attr].nunique()
    return diversity >= l

def check_t_closeness(partition, t):
    # 計算敏感屬性的分佈
    global_dist = data[sensitive_attr].value_counts(normalize=True)
    partition_dist = partition[sensitive_attr].value_counts(normalize=True)

    # 計算區塊與全體的距離
    distance = entropy(partition_dist, global_dist, base=2)
    return distance <= t

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
        # 檢查l-diversity 和 t-closeness
        if check_l_diversity(lhs, l) and check_l_diversity(rhs, l) and check_t_closeness(lhs, t) and check_t_closeness(rhs, t):
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
print(result)
result.to_csv('anon_k=' + str(k) + '_l=' + str(l) + '_t=' + str(t) + '.csv', index=False)

