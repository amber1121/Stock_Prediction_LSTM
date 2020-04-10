import numpy as np
import matplotlib.pyplot as plt

def draw_result(real_price, prediction_price,company_name):
    plt.figure()
    plt.plot(real_price)
    plt.plot(prediction_price)
    plt.title('Prediction vs Real Stock Price')
    plt.ylabel('Price')
    plt.xlabel('Days')
    plt.legend(['Prediction', 'Real'], loc='upper left')
    plt.savefig(company_name)
    plt.show()

if __name__ == "__main__":
    real_price = np.load('Real.npy')
    prediction_price = np.load('Prediction.npy')
    draw_result(real_price, prediction_price,'FB_2_result.png')
    