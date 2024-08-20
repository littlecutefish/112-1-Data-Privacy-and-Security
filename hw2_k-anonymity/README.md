## Hw2 - k-anonymity

參考：https://github.com/kaylode/k-anonymity/tree/main

1. 透過 anonymize.py 產出隱私保護的.CSV檔案
```
# terminal, k可隨意改變
python anonymize.py --method=mondrian --k=10 --dataset=adult
```

2. 執行 compare.py 計算 Misclassification Error, Accuracy, Precision, Recall, AUC 指標

    (範例1) 比較 origin & k = 30
   
    <img width="364" alt="截圖 2024-08-20 晚上11 44 43" src="https://github.com/user-attachments/assets/3e97a2b1-c010-410f-91ee-1366a5956157">

    (範例2) 比較 origin & k = 100
   
   <img width="374" alt="截圖 2024-08-20 晚上11 44 00" src="https://github.com/user-attachments/assets/93a0ded7-537d-4e67-b3c0-7cb6e473e4d0">


    
