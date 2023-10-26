import numpy as np
import pandas as pd

class Curve:
    def __init__(self, name):
        self.name = name
        self.data_path = rf"CurvesData/Curve_{name}.csv"
        self.pillar_dates, self.pillars = self.GetPillarsFromFiles()
        self.pillar_timestamps = self.pillar_dates.map(lambda x: pd.Timestamp(x).timestamp())

    def GetPillarsFromFiles(self):
        data_df = pd.read_csv(self.data_path)
        return data_df["date"], data_df["discount_factor"]

    def DF(self, t):
        return np.interp(t, self.pillar_timestamps, self.pillars)

name = "EUR"
curve = Curve(name)
t = "2/4/2020"
df = curve.DF(pd.Timestamp(t).timestamp())
print(df)