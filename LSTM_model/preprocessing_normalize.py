import pandas as pd
import numpy as np

class DataProcessing:
    def __init__(self, file, ratio):
        self.ratio = ratio
        self.file = pd.read_csv(file)

    def split_partion_size(self, ratio):
        return int(ratio * len(self.file))


    def split_data(self, seq_len):
        x_train, y_train = self.gen_train(11)
        x_test, y_test = self.gen_test(11)
        return x_train, y_train, x_test, y_test

    def gen_train(self, seq_len):
        """
        Generates training data
        :param seq_len: length of window
        :return: X_train and Y_train
        """
        return self.gen_data(seq_len, self.get_stock_train())

    def get_stock_train(self):
        return  self.file[0: self.split_partion_size(self.ratio)]

    def gen_test(self, seq_len):
        """
        Generates test data
        :param seq_len: Length of window
        :return: X_test and Y_test
        """
        return self.gen_data(seq_len, self.get_stock_test())

    def get_stock_test(self):
        return self.file[self.split_partion_size(self.ratio):]

    def gen_data(self, seq_len, stock_file):
        drop_column_stock_file = self.drop_column(stock_file)
        x_data = self.format_x_data(self.normalize(drop_column_stock_file), seq_len)
        y_data = self.format_y_data(drop_column_stock_file, seq_len)
        return x_data, y_data

    def drop_column(self, df):
        return df.drop(['Date'], axis = 1)

    def normalize(self, df):
        return df.pct_change(fill_method ='ffill')[1:]

    def format_x_data(self, df, seq_len):
        INPUT_DIM = 5
        x_data = []
        for start in range((len(df) // seq_len) * seq_len - seq_len - 1):
            end_date = start + seq_len
            x = np.array(df.iloc[start: end_date])
            x_data.append(x)
        x_train_data = np.array(x_data)
        return x_train_data.reshape((len(x_train_data), seq_len, INPUT_DIM))

    def format_y_data(self, df, seq_len):    
        y_data = [0]
        for start in range((len(df) // seq_len) * seq_len - seq_len - 1):
            end_date = start + seq_len
            yesterday_y = np.array([df.iloc[end_date, self.open_attribute_index()]], np.float64)
            next_date = end_date + 1
            y = np.array([df.iloc[next_date, self.open_attribute_index()]], np.float64)
            y_data.append(self.trend(yesterday_y, y))
        return np.array(y_data[0:-1])

    def open_attribute_index(self):
        return 0

    def trend(self,yesterday_y, current_y):
        raise_percent = (current_y - yesterday_y) / yesterday_y
        threshold = 0.0045
        if raise_percent > threshold:
            return 2
        elif raise_percent <= threshold and raise_percent >= -threshold:
            return 1
        elif raise_percent < -threshold:
            return 0 

if __name__ == "__main__":
    process = DataProcessing('FB.csv', 0.8)
    x, y = process.gen_train(10)
    print(len(x))
    print(len(y))