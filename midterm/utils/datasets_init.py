from .types import Dataset

def get_dataset_params(name):
    if name == Dataset.ADULT:
        QI_INDEX = [1, 2, 3, 4, 5, 6, 7, 8]
        target_var = 'salary-class'
        IS_CAT = [True, False, True, True, True, True, True, True]
        max_numeric = {"age": 50.5}
    elif name == Dataset.INFORMS:
        QI_INDEX = [3, 4, 6, 13, 16]
        target_var = "poverty"
        IS_CAT = [True, True, True, True, False]
        max_numeric = {"DOBMM": None, "DOBYY": None, "RACEX": None, "EDUCYEAR": None, "income": None}
    else:
        print(f"Not support {name} dataset")
        raise ValueError
    return {
        'qi_index': QI_INDEX,
        'is_category': IS_CAT,
        'target_var': target_var,
        'max_numeric': max_numeric
    }