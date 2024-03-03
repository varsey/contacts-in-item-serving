import os
import pickle
from datetime import datetime
from typing import Optional, List

import GPUtil
import lightgbm as lgb
from pandas import DataFrame as pdf
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score
from lib.src.s3_uploader import UploaderS3


class Trainer:
    def __init__(self, pkl_dir: str, file_catalog: dict, cat_feats_list: Optional[List]):
        self.cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
        self.pkl_dir = ''.join([os.getcwd(), pkl_dir])
        self.file_catalog = file_catalog
        self.cat_feats_list = cat_feats_list
        self.device_type = 'gpu' if len(GPUtil.getAvailable()) > 0 else 'cpu'
        print(f'Device type for training: {self.device_type}')
        self.s3_client = UploaderS3()

    def split_data(self, x, y):
        x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True,
                                                            test_size=0.2, random_state=42, stratify=y)
        return x_train, x_test, y_train, y_test

    def train_on_data(self, x, y) -> None:
        """Тренирует три модели одну за одной. Сохраняет результат в pkl файлы"""
        x_train, x_test, y_train, y_test = self.split_data(x, y)
        print('Training models')
        str_keys = ''.join(self.file_catalog.keys())
        if 'lgb' in str_keys:
            self.train_lgb(x_train, x_test, y_train, y_test, x, y)
        if 'lr' in str_keys:
            self.train_lr(x_train, x_test, y_train, y_test, x, y)

        return None

    @staticmethod
    def get_roc_auc(model, x_test, y_test, y_pred) -> float:
        accuracy = accuracy_score(y_test, y_pred)
        print(f'Accuracy: {accuracy:.3f}')

        y_pred_proba = model.predict_proba(x_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        print(f'ROC-AUC score: {roc_auc_score(y_test, y_pred_proba):.3f}')

        return roc_auc

    def train_lgb(self, x_train: pdf, x_test: pdf, y_train: pdf, y_test: pdf, x: pdf, y: pdf) -> None:
        clf = lgb.LGBMClassifier(
            objective='binary',
            metric='auc',
            boosting_type='gbdt',
            class_weight='balanced',
            lambda_l1=1,
            lambda_l2=0.03,
            n_estimators=100,
            num_leaves=256,
            verbose=-1,
            n_jobs=-1,
            seed=42,
            categorical_feature=self.cat_feats_list,
            device=self.device_type,
        )
        pre_clf_model = clf.fit(x_train, y_train)

        y_pred = pre_clf_model.predict(x_test)
        roc_auc = self.get_roc_auc(clf, x_test, y_test, y_pred)

        print('Training final model on full ds...')
        clf_model = clf.fit(x, y)
        pickle.dump(clf_model, open(f'{self.pkl_dir}{self.file_catalog["lgbm_model_file"]}', 'wb'))
        self.s3_client.upload_to_s3(
            f'{self.pkl_dir}{self.file_catalog["lgbm_model_file"]}',
            f'lgbm_model_file_{roc_auc:.5f}_{datetime.now()}.pkl'
        )

        return None

    def train_lr(self, x_train: pdf, x_test: pdf, y_train: pdf, y_test: pdf, x: pdf, y: pdf) -> None:
        lr_model = LogisticRegression(
            penalty='l2',
            solver='saga',
            random_state=42,
            n_jobs=-1,
            max_iter=50,
        )
        pre_lr_model= lr_model.fit(x_train, y_train)
        y_pred = pre_lr_model.predict(x_test)
        roc_auc = self.get_roc_auc(pre_lr_model, x_test, y_test, y_pred)

        print('Training final model on full ds...')
        lr_model = lr_model.fit(x, y)
        pickle.dump(lr_model, open(f'{self.pkl_dir}{self.file_catalog["lr_model_file"]}', 'wb'))
        self.s3_client.upload_to_s3(
            f'{self.pkl_dir}{self.file_catalog["lr_model_file"]}',
            f'lr_model_file_{roc_auc:.5f}_{datetime.now()}.pkl'
        )

        return None

