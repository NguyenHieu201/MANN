import pandas as pd
import json

config_path = "./expconfig.json"
config = json.load(open(config_path))

def cutting_dataframe(path, start_day, finish_day):
    df = pd.read_csv(path)
    result = df[(df['datetime'] > start_day) & (df['datetime'] < finish_day)]
    return result
