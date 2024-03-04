import os
import sys
import logging

import pandas as pd


class DataLoader:
    train_csv = 'train.csv' if os.getenv('TRAIN_DATA_FILENAME') is None else os.getenv('TRAIN_DATA_FILENAME')
    val_csv = 'val.csv' if os.getenv('VAL_DATA_FILENAME') is None else os.getenv('VAL_DATA_FILENAME')

    def __init__(self, validate: bool = True):
        self.validate = validate
        self.logger = self._get_logger()
        self.data_dir = os.getenv('DATA_ROOT')
        self.user = os.getenv('USER')

    @staticmethod
    def _get_logger():
        logger = logging.getLogger()
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        return logger

    def load_train_data(self) -> pd.DataFrame:
        path = os.path.join(self.data_dir, self.train_csv)
        if not os.path.exists(path):
            self.logger.info(f'Файл {path} не найден')
            raise ValueError('train не найден')
        df = pd.read_csv(path)

        return df

    def load_test_data(self) -> pd.DataFrame:
        path = os.path.join(self.data_dir, self.val_csv)
        if not os.path.exists(path):
            self.logger.info(f'Файл {path} не найден')
            raise ValueError('test не найден')
        df = pd.read_csv(path)
        if 'is_bad' in df:
            del df['is_bad']

        return df
