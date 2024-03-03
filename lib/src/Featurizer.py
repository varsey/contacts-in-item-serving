import pandas as pd
from .FeatureGenerator import FeatureGenerator


class Featurzer:
    @staticmethod
    def categorize_data(x_stacked, x_vect):
        print('Setting categories to data...')
        x = pd.DataFrame.sparse.from_spmatrix(x_stacked)

        cat_feats_list = [x for x in range(x_stacked.shape[1] - (x_stacked.shape[1] - x_vect.shape[1]), x_stacked.shape[1])]
        for col in cat_feats_list:
            x[col] = x[col].astype(int)
            x[col] = pd.Categorical(x[col])

        return x, cat_feats_list

    @staticmethod
    def add_features(data: pd.DataFrame) -> pd.DataFrame:
        print('Adding features to original dataset')
        data['is_phone_present'] = data["text_cleaned"].apply(lambda x: FeatureGenerator.parse_tel(x))
        data['has_at'] = data["text_cleaned"].apply(lambda x: FeatureGenerator.has_at(x))
        data['text_cleaned_str'] = FeatureGenerator.calculate_length_lst(data=data, field="text_cleaned")
        return data

    def featurized_data(self, data: pd.DataFrame) -> pd.DataFrame:
        return self.add_features(data)
