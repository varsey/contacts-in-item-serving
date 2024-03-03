from typing import Optional

import pandas as pd
from lib.src.preprocessor import Preprocessor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import roc_auc_score, accuracy_score


class Labeleler(Preprocessor):
    def __init__(self, n_features: int, vocabulary_file: str, val_data: pd.DataFrame):
        super().__init__(n_features, vocabulary_file)
        self.labeling_model: LogisticRegression = Optional[LogisticRegression]
        self.val_data: pd.DataFrame = val_data
        self.lv = self.v.init_vectorizer(load_vocab=True)

    def train_labeler(self):
        """Trains lr model on val data for relabeling train data"""
        text_data = self.val_data['text_cleaned'].to_list()
        pre_y = self.val_data.is_bad.to_list()

        pre_x_vect = self.lv.fit_transform(text_data)

        selector = SelectKBest(chi2, k=self.n_features)
        pre_x_vect_sel = selector.fit_transform(pre_x_vect, pre_y)

        pre_x_train, pre_x_test, pre_y_train, pre_y_test = train_test_split(pre_x_vect_sel, pre_y,
                                                                            shuffle=True, test_size=0.2,
                                                                            stratify=pre_y)

        pre_model = LogisticRegression(
            penalty='l1',
            class_weight='balanced',
            solver='saga',
            random_state=42,
            n_jobs=-1,
            max_iter=4,
        )

        pre_model_fit = pre_model.fit(pre_x_train, pre_y_train)

        y_pred = pre_model_fit.predict(pre_x_test)

        accuracy = accuracy_score(pre_y_test, y_pred)
        print(f'Accuracy: {accuracy:.3f}')

        y_pred_proba = pre_model_fit.predict_proba(pre_x_test)[:,1]
        roc_auc_score(pre_y_test, y_pred_proba).__round__(3)

        self.labeling_model = pre_model.fit(pre_x_vect_sel, pre_y)
        return None

    def label_train(self, x_final, train_data):
        self.train_labeler()
        if not self.labeling_model:
            return x_final, train_data.is_bad
        else:
            print(x_final.shape)
            relabeled_y_proba = self.labeling_model.predict_proba(x_final)[:, 1]

        relabeled_train = pd.concat(
            [
                train_data.text_cleaned.reset_index(drop=True),
                pd.DataFrame(relabeled_y_proba)
            ],
            axis=1
        )

        new_train_f = relabeled_train[(relabeled_train[0] > 0.8) | (relabeled_train[0] < 0.6)]

        post_x_vect = self.lv.fit_transform(new_train_f['text_cleaned'].to_list())
        post_y = [1 if x > 0.7 else 0 for x in new_train_f[0].to_list()]

        return post_x_vect, post_y
