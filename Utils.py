import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Get close data from csv file
def get_data(folder, src_name):
    path = os.path.join(folder, f"{src_name}.csv")
    df = pd.read_csv(path)
    data = df.close.to_numpy()
    return data


def time_series_processing(data, mode, setting):
    match mode:
        case "one-day":
            n_sample = data.shape[0]
            seq_len = setting["seq_len"]
            x = [data[i : i+seq_len] for i in range(0, n_sample - seq_len)]
            y = [data[i] for i in range(seq_len, n_sample)]
            return {
                "X": np.array(x, dtype=np.float32),
                "Y": np.array(y, dtype=np.float32).reshape(-1, 1)
            }
            
        case "test":
            seq_len = 22
            horizon = 3
            n_sample = data.shape[0]
            x = [data[i : i+seq_len] for i in range(0, n_sample-seq_len)]
            y = [data[i : i+horizon] for i in range(seq_len, n_sample-horizon)]
            n_sample = min(len(x), len(y))
            x = x[:n_sample]
            y = y[:n_sample]
            return {
                "X": np.array(x, dtype=np.float32),
                "Y": np.array(y, dtype=np.float32)
            }            


def preprocessing(data, name, test_ratio, mode, setting):
    train, test = train_test_split(data, test_size=test_ratio, shuffle=False)

    train = time_series_processing(data=train, mode=mode, setting=setting)
    test = time_series_processing(data=test, mode=mode, setting=setting)
    # Scale data
    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    x_scaler.fit(train["X"])
    y_scaler.fit(train["Y"])
    train["X"] = x_scaler.transform(train["X"])
    train["Y"] = y_scaler.transform(train["Y"])
    test["X"] = x_scaler.transform(test["X"])
    test["Y"] = y_scaler.transform(test["Y"])

    return {
        "name": name,
        "train": train,
        "test": test,
        "x_scaler": x_scaler,
        "y_scaler": y_scaler
    }