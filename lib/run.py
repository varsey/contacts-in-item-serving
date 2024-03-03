import os
import sys
import logging

import pandas as pd


class Test:
    train_csv = 'train.csv'
    val_csv = 'val.csv'
    test_csv = 'test_data.csv'

    def __init__(self, debug: bool = True):
        self.debug = debug
        self.logger = self._get_logger()
        self.data_dir = os.getenv('DATA_ROOT')
        self.test_data_dir = os.getenv('TEST_DATA_ROOT')
        self.user = os.getenv('USER')

    @staticmethod
    def _get_logger():
        logger = logging.getLogger()
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        return logger

    def train_data(self) -> pd.DataFrame:
        path = os.path.join(self.data_dir, self.train_csv)
        if not os.path.exists(path):
            self.logger.info(f'Файл {path} не найден')
            path = os.path.join(self.test_data_dir, self.train_csv)
        if not os.path.exists(path):
            self.logger.info(f'Файл {path} не найден')
            raise ValueError('train не найден')
        df = pd.read_csv(path)

        return df

    def val_data(self) -> pd.DataFrame:
        path = os.path.join(self.data_dir, self.val_csv)
        if not os.path.exists(path):
            self.logger.info(f'Файл {path} не найден')
            path = os.path.join(self.test_data_dir, self.val_csv)
        if not os.path.exists(path):
            self.logger.info(f'Файл {path} не найден')
            raise ValueError('val не найден')
        df = pd.read_csv(path)

        return df

    def test_data(self) -> pd.DataFrame:
        path = os.path.join(self.test_data_dir, self.val_csv if self.debug else self.test_csv)
        if not os.path.exists(path):
            self.logger.info(f'Файл {path} не найден')
            raise ValueError('test не найден')
        df = pd.read_csv(path)
        if 'is_bad' in df:
            del df['is_bad']

        return df



