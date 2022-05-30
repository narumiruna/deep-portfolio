"""
generate data for training, validation and testing
"""
from keras.preprocessing.sequence import TimeseriesGenerator
import numpy as np

def data_split(data, data_len=50, val_ratio=0.2, test_ratio=0.1):
    """
    generate data for training, validation and testing
    """
    generator = TimeseriesGenerator(data=data,
                                    targets=range(data.shape[0]),
                                    length=data_len,
                                    batch_size=1,
                                    stride=1)
    x_all = []
    for i in generator:
        x_all.append(i[0][0])
    x_all = np.array(x_all)
    train_end = int(len(x_all) * (1 - (val_ratio + test_ratio)))
    val_end = int(len(x_all) * (1 - test_ratio))
    x_train = x_all[:train_end]
    x_val = x_all[train_end:val_end]
    x_test = x_all[val_end:]
    return x_train, x_val, x_test
