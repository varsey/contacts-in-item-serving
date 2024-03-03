import os
import pickle
import logging

import pandas as pd
from lib.trainer import Trainer
from lib.src.Preprocessor import Preprocessor
from lib.src.TransformerTrainer import TransformerTrainer

PKL_DIR = '/lib/pkl/'


class Task1:
    # Каталог сериализированных файлов
    model_files = {
        'lgbm_model_file': 'lgbm_model.pkl',
        'lr_model_file': 'lr_model.pkl',
    }
    n_features = 5000
    vocabulary_file = f'new_vocab_{n_features}.pkl'

    def __init__(self):
        self.pre_path = ''.join([os.getcwd(), PKL_DIR])
        os.makedirs(self.pre_path, exist_ok=True)
        self.vocab_path = f'{self.pre_path}{self.vocabulary_file}'
        self.pp = Preprocessor(self.n_features, self.vocab_path)
        self.tt = TransformerTrainer()
        self.loaded_models = self.load_models(self.model_files)
        self.loaded_tt = self.tt.load_model()

    def load_pickle(self, tag: str) -> pickle:
        """Wrapper для загрузки пикл-файлов по метке"""
        path = f'{self.pre_path}{self.model_files[tag]}'
        with open(path, "rb") as file:
            model = pickle.load(file)
        return model

    def load_models(self, catalog: dict) -> dict:
        loaded_models = {}
        for model_name in catalog.keys():
            loaded_models[model_name] = self.load_pickle(model_name)
        return loaded_models

    def predict(self, test_data: pd.DataFrame, train_data: pd.DataFrame, force_retrain: bool = False) -> float:

        is_file_exists = [os.path.exists(f'{self.pre_path}{self.model_files[tag]}') for tag in self.model_files.keys()]
        if force_retrain or not all(is_file_exists):
            print("Started TRAIN data preprocessing")
            train_x, train_data_a = self.pp.vectorizing_pipeline(train_data)

            print("Started models training")
            t = Trainer(pkl_dir=PKL_DIR, file_catalog=self.model_files, cat_feats_list=None)
            t.train_on_data(train_x, train_data_a.is_bad)

            data_train, data_test, _, _ = t.split_data(train_data_a, train_data_a.is_bad)
            dd = self.tt.create_ddataset(data_train, data_test)
            self.tt.train(
                self.tt.encode_ddataset(dd)
            )

        if not os.path.exists(self.vocab_path):
            print("Creating vocabulary")
            self.pp.vectorizing_pipeline(train_data)

        print("Started TEST data preprocessing")
        data_to_predict_on = self.pp.clean_df(test_data)
        test_x = self.pp.vectorize_to_nfeatures(data_to_predict_on)

        y_pred = 0
        for model in self.loaded_models.values():
            print(model)
            y_pred += model.predict_proba(test_x)[:, 1]

        print('tensorflow start')
        tt_data_to_predict_on = self.tt.prepare_validation(data_to_predict_on[['text_cleaned']])

        tt_preds = self.loaded_tt.predict(tt_data_to_predict_on["validation"]).predictions[:, 1]
        y_pred += self.tt.sigmoid(tt_preds)

        return y_pred / (len(self.loaded_models) + 1)
