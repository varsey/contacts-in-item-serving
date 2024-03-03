import os

import pandas as pd
import nlpaug.augmenter.word as naw

PPDB_DIR = '/ppdb/'
MODEL_NAME = 'ppdb-1.0-s-lexical'


class Augmenter:
    def __init__(self):
        self.aug = naw.SynonymAug(
            aug_src='ppdb',
            model_path=''.join([os.getenv('HOME'), PPDB_DIR, MODEL_NAME])
        )

    def augment(self, data: pd.DataFrame):
        underrepresented = data[data.category.isin(['Работа', 'Услуги', 'Хобби и отдых'])]
        underrepresented.loc[:, "text_cleaned"] = underrepresented.loc[:, "text_cleaned"].apply(lambda x: self.aug.augment(x)[0]).copy()
        return pd.concat([data, underrepresented]).drop_duplicates()
