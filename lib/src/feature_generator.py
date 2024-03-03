import regex as re

import pandas as pd


class FeatureGenerator:
    @staticmethod
    def parse_tel(text: str) -> int:
        candidate = ''.join(re.findall(r'\d+', text))[-11:]
        return 1 if len(candidate) == 11 and candidate[0] in ['7', '8', '9'] or candidate[1] == '9' else 0

    @staticmethod
    def has_at(text: str) -> bool:
        return True if '@' in text else False

    @staticmethod
    def calculate_length_lst(data: pd.DataFrame, field: str) -> list:
        return data[field].str.len()
