import pickle
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_extraction.text import TfidfVectorizer


class Vectorizer:
    def __init__(self, n_features: int, vocabulary_file: str):
        self.n_features = n_features
        self.vocabulary_file = vocabulary_file

    def init_vectorizer(self, load_vocab: bool = False):
        if load_vocab:
            with open(self.vocabulary_file, "rb") as file:
                vocab = pickle.load(file)
            return TfidfVectorizer(vocabulary=vocab)
        return TfidfVectorizer()

    def save_kbest_features_vocab(self, fiited_vectorizer, x, y) -> None:
        selector = SelectKBest(chi2, k=self.n_features)
        selector.fit_transform(x, y)

        print('Saving vocabulary')
        feature_names = fiited_vectorizer.get_feature_names_out()[selector.get_support()]
        pickle.dump(obj=feature_names, file=open(self.vocabulary_file, 'wb'))

        return None
