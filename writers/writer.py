import logging
from pathlib import Path

import pandas as pd
import csv


class Writer:
    def __init__(self, csv_file: Path, index_row: str):
        self.csv_file = csv_file
        self.index_row = index_row
        self.data = None
        self.headers = None
        self.saved_result = None
        self.saved_output = None
        self.__load_data()

    def write(self, info: dict):
        idx = info[self.index_row]
        if idx in self.data.index:
            logging.error("Index already exists, double check first!\n{}".format(info))
            return
        self.data = self.data.append(pd.DataFrame(info, index=[idx]))
        try:
            with self.csv_file.open('a',encoding="utf-8", newline='\n') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self.headers)
                writer.writerow(info)
        except Exception as e:
            self.data = self.data.drop(idx)
            raise e

    def update_or_write(self, info: dict):
        idx = info[self.index_row]
        if idx not in self.data.index:
            self.write(info)
            return
        new_info = info.copy()
        new_info.pop(self.index_row)
        self.data.loc[idx, new_info.keys()] = new_info.values()
        self.data.to_csv(self.csv_file, index_label=self.index_row)

    def get_row(self, idx, non_existing=None):
        try:
            records = self.data.loc[idx].to_dict()
        except KeyError:
            return non_existing
        records[self.index_row] = idx
        return records

    def get_rows_by_column(self, query_fields):
        query = ' & '.join([f'{k}=="{v}"' for k, v in query_fields.items()])
        self.saved_result = self.data.query(query).copy()
        self.saved_result['race_id'] = self.saved_result.index
        self.saved_output = self.saved_result.to_dict('records')
        return self.saved_output

    def __load_data(self):
        try:
            self.data = pd.read_csv(self.csv_file).set_index(self.index_row)
        except FileNotFoundError:
            raise FileNotFoundError("Couldn't find the metadata file {}".format(self.csv_file))

        self.data['last_update'] = pd.to_datetime(self.data['last_update'])
        self.headers = [self.index_row] + list(self.data.columns)
