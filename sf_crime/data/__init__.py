import os
import datetime
import pandas as pd
import sklearn.preprocessing

from util import INPUT_PATH, OUTPUT_PATH


def parse_time(timestamp):
    dt = datetime.datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
    return dt.hour, dt.day, dt.month, dt.year


class DataHandler():
    def __init__(self):
        self.train_df = pd.read_csv(os.path.join(INPUT_PATH, "train.csv"))
        self.test_df = pd.read_csv(os.path.join(INPUT_PATH, "test.csv"))

        self.data_df = self.train_df.append(self.test_df).reset_index(drop=True)
        self.data_df = self.data_df.drop(["Id", "Category", "Descript", "Resolution"], axis=1)

        sklearn.preprocessing.scale(self.data_df[["X", "Y"]], copy=False)
        series = self.data_df["Dates"].apply(parse_time)  # [(H, d, M, Y), (H, d, M, Y), ...]
        self.data_df["Hour"], self.data_df["Day"], self.data_df["Month"], self.data_df["Year"] = zip(*series)  # [[H, ...], [d, ...], [M, ...], [Y, ...]]

        self.data_df["DayOfWeek"], self.indices_dow = pd.factorize(self.data_df["DayOfWeek"])
        self.data_df["PdDistrict"], self.indices_pd = pd.factorize(self.data_df["PdDistrict"])

        self.data_df = self.data_df.drop(["Dates", "Address"], axis=1)

        self.train = self.data_df[:self.train_df.shape[0]].values
        self.test = self.data_df[self.train_df.shape[0]:].values
        self.target, self.indices_cat = pd.factorize(self.train_df["Category"])
        self.ids = self.test_df.index.values


    def save_submission(self, prob, name="submission"):
        id_df = pd.DataFrame({"Id": self.ids})
        prob_df = pd.DataFrame(prob, columns = self.indices_cat)
        p = os.path.join(OUTPUT_PATH, "%s.csv.gz" % name)
        id_df.join(prob_df).sort_index(axis=1).to_csv(p, index=False, compression="gzip")
