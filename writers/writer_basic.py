from typing import Optional

import pandas as pd

class WriterBasic:
    def __init__(self, dir):
        self.dir = dir

    def save(self, file_name: str, data: pd.DataFrame) -> bool:
        if len(data) == 0:
             return False
        data.to_csv(self.dir / ('{}.csv'.format(file_name)), index=False)
        return True

    def read(self, file_name: str) -> Optional[pd.DataFrame]:
        try:
            data = pd.read_csv(self.dir / ('{}.csv'.format(file_name)))
        except FileNotFoundError:
            return None
        return data
