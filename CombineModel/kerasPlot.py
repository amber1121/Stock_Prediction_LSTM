from keras.utils import plot_model


class KerasPlot:
    def __init__(self):
        pass

    @staticmethod
    def draw(model, file_name):
        plot_model(model, to_file=file_name, show_shapes=True)