import numpy as np
import os
from PathObject import PathObject
class Filter:

    def __init__(self, han_x_train_dir, lstm_x_train_dir, lstm_y_train_dir, han_x_test_dir,
    lstm_x_test_dir, lstm_y_test_dir):
        self.han_x_train_dir = han_x_train_dir
        self.lstm_x_train_dir = lstm_x_train_dir
        self.lstm_y_train_dir = lstm_y_train_dir
        self.han_x_test_dir = han_x_test_dir
        self.lstm_x_test_dir = lstm_x_test_dir
        self.lstm_y_test_dir = lstm_y_test_dir

    def extract_train_company_name(self, dir_path):
        result = []
        for company in os.listdir(dir_path):
            company_name = company[:-12]
            result.append(company_name)
        return result

    def extract_test_company_name(self, dir_path):
        result = []
        for company in os.listdir(dir_path):
            company_name = company[:-11]
            result.append(company_name)
        return result

    def intersection_company_name(self):
        han_x_train_company = self.extract_train_company_name(self.han_x_train_dir)
        lstm_x_train_company = self.extract_train_company_name(self.lstm_x_train_dir)
        lstm_y_train_company = self.extract_train_company_name(self.lstm_y_train_dir)
        han_x_test_company = self.extract_test_company_name(self.han_x_test_dir)
        lstm_x_test_company = self.extract_test_company_name(self.lstm_x_test_dir)
        lstm_y_test_company = self.extract_test_company_name(self.lstm_y_test_dir)
        res = set(han_x_train_company) & set(lstm_x_train_company) & set(lstm_y_train_company) & set(han_x_test_company) & set(lstm_x_test_company) & set(lstm_y_test_company)
        return list(res)

    def get_path_objects(self):
        result = []
        for company in self.intersection_company_name():
            han_x_train_path = self.han_x_train_dir + company + '_x_train.npy'
            lstm_x_train_path = self.lstm_x_train_dir + company + '_x_train.npy'
            lstm_y_train_path = self.lstm_y_train_dir + company + '_y_train.npy'
            han_x_test_path = self.han_x_test_dir + company + '_x_test.npy'
            lstm_x_test_path = self.lstm_x_test_dir + company + '_x_test.npy'
            lstm_y_test_path = self.lstm_y_test_dir + company + '_y_test.npy'
            path_object = PathObject(han_x_train_path, lstm_x_train_path, lstm_y_train_path,
            han_x_test_path, lstm_x_test_path, lstm_y_test_path)
            result.append(path_object)
        return result
        
if __name__ == "__main__":
    han_x_train_dir = './data/han_x_train/'
    lstm_x_train_dir = './data/lstm_x_train/'
    lstm_y_train_dir = './data/lstm_y_train/'
    han_x_test_dir = './data/han_x_test/'
    lstm_x_test_dir = './data/lstm_x_test/'
    lstm_y_test_dir = './data/lstm_y_test/'

    filter_object = Filter(han_x_train_dir, lstm_x_train_dir, lstm_y_train_dir, han_x_test_dir,
    lstm_x_test_dir, lstm_y_test_dir)

    path_objects = filter_object.get_path_objects(PathObject)
    print(len(path_objects))
    
    

    
    