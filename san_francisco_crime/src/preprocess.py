# Preprocesser
import math
import numpy as np
import pandas as pd

def make_one_hot_vector(dims, hot):
    v = np.zeros(dims, dtype=np.float32)
    v[hot] = 1.
    return v

class SanFranciscoCrimeData(object):
    # Dates
    # Category (TARGET)
    # Descript (Only train.csv)
    # DayOfWeek
    # PdDistrict
    # Resolution (Only train.csv)
    # Address
    # X (Longitude)
    # Y (Latitude)
    train_data_name = '../data/train.csv'
    test_data_name = '../data/test.csv'

    # DayOfWeek
    day_dict = {'Sunday': 0,
                'Monday': 1,
                'Tuesday': 2,
                'Wednesday': 3,
                'Thursday': 4,
                'Friday': 5,
                'Saturday': 6}

    def __init__(self):
        self.train_df = pd.read_csv(self.train_data_name)
        self.test_df = pd.read_csv(self.test_data_name)

        # PdDistrict
        pdd = list(set(self.train_df['PdDistrict']))
        pdd.sort()
        self.pd_dict = {p:i for i, p in enumerate(pdd)}

        # Category
        cat = list(set(self.train_df['Category']))
        cat.sort()
        self.cat_dict = {c:i for i, c in enumerate(cat)}


    # DayOfWeek
    def get_day_of_week_vector(self, day):
        return make_one_hot_vector(7, self.day_dict[day])

    def split_day_of_week(self):
        return [pd.DataFrame(self.train_df[self.train_df['DayOfWeek']==d]) for d in day_dict.keys()]

    # PdDistrict
    def get_pd_district_vector(self, pd):
        return make_one_hot_vector(len(self.pd_dict), self.pd_dict[pd])

    def split_pd_district(self):
        return [pd.DataFrame(self.train_df[self.train_df['PdDistrict']==p]) for p in pd_dict.keys()]

    # Category
    def get_category_vector(self, cat):
        return make_one_hot_vector(len(self.cat_dict), self.cat_dict[cat])

