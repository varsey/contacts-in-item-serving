import pandas as pd
from scipy.sparse import hstack
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder


class Stacker:
    def __init__(self):
        self.oh_encoder = OneHotEncoder(handle_unknown='ignore')
        self.kb_encoder = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')

    def stack_extra_feats(self, x_vect, data: pd.DataFrame):
        """Parametrize field names"""
        print('Stacking features')
        is_phone_present = data['is_phone_present'].to_list()
        x_is_phone_present = pd.DataFrame({'is_phone_present': is_phone_present})

        is_at_present = data['is_phone_present'].to_list()
        x_is_at_present = pd.DataFrame({'is_at_present': is_at_present})

        x_category = self.oh_encoder.fit_transform(data[['category']])
        x_subcategory = self.oh_encoder.fit_transform(data[['subcategory']])

        x_binned = self.kb_encoder.fit_transform(data[['text_cleaned_str']])
        x_encoded_len_bins = self.kb_encoder.fit_transform(x_binned)

        return hstack([x_vect, x_is_phone_present, x_is_at_present, x_category, x_subcategory, x_encoded_len_bins])