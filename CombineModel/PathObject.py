class PathObject:
    result = []
    def __init__(self, han_x_train_file, lstm_x_train_file, lstm_y_train_file, han_x_test_file,
    lstm_x_test_file, lstm_y_test_file):
        self.han_x_train_file = han_x_train_file
        self.lstm_x_train_file = lstm_x_train_file
        self.lstm_y_train_file = lstm_y_train_file
        self.han_x_test_file = han_x_test_file
        self.lstm_x_test_file = lstm_x_test_file
        self.lstm_y_test_file = lstm_y_test_file
     
    def get_han_x_train(self):
        return self.han_x_train_file
    
    def get_lstm_x_train(self):
        return self.lstm_x_train_file
    
    def get_lstm_y_train(self):
        return self.lstm_y_train_file
    
    def get_han_x_test(self):
        return self.han_x_test_file
    
    def get_lstm_x_test(self):
        return self.lstm_x_test_file
    
    def get_lstm_y_test(self):
        return self.lstm_y_test_file
        
