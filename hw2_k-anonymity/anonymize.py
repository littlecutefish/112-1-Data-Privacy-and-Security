from utils.types import AnonMethod
import os
import argparse
import numpy as np
import pandas as pd
from metrics import NCP, DM, CAVG

from algorithm import (
        k_anonymize,
        read_tree)
from utils.datasets_init import get_dataset_params
from utils.data import read_raw, write_anon, numberize_categories

parser = argparse.ArgumentParser('K-Anonymize')
parser.add_argument('--method', type=str, default='mondrian',
                    help="K-Anonymity Method")
parser.add_argument('--k', type=int, default=2,
                    help="K-Anonymity")
parser.add_argument('--dataset', type=str, default='adult',
                    help="Dataset to anonymize")


class Anonymizer:
    def __init__(self, args):
        self.method = args.method
        assert self.method in ["mondrian"]
        self.k = args.k
        self.data_name = args.dataset
        self.csv_path = args.dataset + '.csv'

        # Dataset path
        self.data_path = os.path.join('data', self.csv_path)

        # Generalization hierarchies path
        self.gen_path = os.path.join('data', 'adult_hierarchies')  # trailing /

        # folder for all results
        res_folder = os.path.join('results')

        # path for anonymized datasets 輸出後的檔案位址
        self.anon_folder = res_folder  # trailing /

        os.makedirs(self.anon_folder, exist_ok=True)

    def anonymize(self):
        # 讀取資料
        data = pd.read_csv(self.data_path, delimiter=';')

        # 第一行的欄位名稱
        ATT_NAMES = list(data.columns)

        data_params = get_dataset_params(self.data_name)
        QI_INDEX = data_params['qi_index']
        IS_CAT2 = data_params['is_category']

        QI_NAMES = list(np.array(ATT_NAMES)[QI_INDEX])
        IS_CAT = [True] * len(QI_INDEX)  # is all cat because all hierarchies are provided
        SA_INDEX = [index for index in range(len(ATT_NAMES)) if index not in QI_INDEX]
        SA_var = [ATT_NAMES[i] for i in SA_INDEX]

        ATT_TREES = read_tree(
            self.gen_path,
            self.data_name,
            ATT_NAMES,
            QI_INDEX, IS_CAT)

        raw_data, header = read_raw(
            'data',
            self.data_name,
            QI_INDEX, IS_CAT)

        anon_params = {
            "name": self.method,
            "att_trees": ATT_TREES,
            "value": self.k,
            "qi_index": QI_INDEX,
            "sa_index": SA_INDEX
        }

        anon_params.update({'data': raw_data})

        print(f"Start anonymizing with {self.method}")
        anon_data, runtime = k_anonymize(anon_params)

        # Write anonymized table
        if anon_data is not None:
            nodes_count = write_anon(
                self.anon_folder,
                anon_data,
                header,
                self.k,
                self.data_name)

        # Normalized Certainty Penalty
        ncp = NCP(anon_data, QI_INDEX, ATT_TREES)
        ncp_score = ncp.compute_score()

        # Discernibility Metric

        raw_dm = DM(raw_data, QI_INDEX, self.k)
        raw_dm_score = raw_dm.compute_score()

        anon_dm = DM(anon_data, QI_INDEX, self.k)
        anon_dm_score = anon_dm.compute_score()

        # Average Equivalence Class

        raw_cavg = CAVG(raw_data, QI_INDEX, self.k)
        raw_cavg_score = raw_cavg.compute_score()

        anon_cavg = CAVG(anon_data, QI_INDEX, self.k)
        anon_cavg_score = anon_cavg.compute_score()

        '''
        1. NCP (Normalized Certainty Penalty): 
            -> NCP 衡量匿名化資料集中確保隱私的程度。其值範圍是 0 到 1，值越低表示資料的隱私保護效果越好。
        
        2. CAVG (Average Equivalence Class Size):
            -> CAVG 衡量匿名化資料集中每個等價類的平均大小。理想情況下，CAVG 的值應該接近 1，表示每個等價類都包含幾乎相同數量的記錄。
        
        3. DM (Discernibility Metric):
            -> DM 衡量匿名化資料集中的可區分性。它表示資料集中不同等價類的數量。數值越低表示資料越不可區分，即資料的隱私保護程度越高。
        '''

        print(f"NCP score (lower is better): {ncp_score:.3f}")
        print(f"CAVG score (near 1 is better): BEFORE: {raw_cavg_score:.3f} || AFTER: {anon_cavg_score:.3f}")
        print(f"DM score (lower is better): BEFORE: {raw_dm_score} || AFTER: {anon_dm_score}")
        print(f"Time execution: {runtime:.3f}s")

        return ncp_score, raw_cavg_score, anon_cavg_score, raw_dm_score, anon_dm_score

def main(args):
    anonymizer = Anonymizer(args)
    anonymizer.anonymize()


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)