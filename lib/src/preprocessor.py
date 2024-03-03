import regex as re

import nltk
import pandas as pd
from pymystem3 import Mystem
from lib.src.vectorizer import Vectorizer
from lib.src.augmenter import Augmenter
pd.options.mode.chained_assignment = None


class Preprocessor:
    flt_chars = [
        ')', '(', ',', '!', ' .', ':', '|', '>', '<', '*', '[', ']', '...', './', '/\n', '\xa0',
        '«', '»', '_', '?', '~', '=', '\\', ';', '\n', '``', "''", '✅', '/n', '/', '..', '.', '`', "'",
        '✔', '⇒', '"', '$', '%', '&', "'", ',', '^', '`', '{', '}',
    ]
    numbers_chars = [(' ноль ', ' 0 '), (' один ', ' 1 '), (' два ', ' 2 '), (' три ', ' 3 '), (' четыре ', ' 4 '),
                     (' пять ', ' 5 '), (' шесть ', ' 6 '), (' семь ', ' 7 '), (' восемь ', ' 8 '), (' девять ', ' 9 '),
                     (' десять ', ' 10 '), (' двадцать ', ' 20 '), (' тридцать ', ' 30 '), (' сорок ', ' 40 '),
                     (' пятьдесят ', ' 50 '), (' шестьдесят ', ' 60 '), (' семьдесят ', ' 70 '),
                     (' восемьдесят ', ' 80 '), (' девяносто ', ' 90 '), (' сто ', ' 100 '),
                     (' двести ', ' 200 '), (' триста ', ' 300 '), (' четыреста ', ' 400 '), (' пятьсот ', ' 500 '),
                     (' шестьсот ', ' 600 '), (' семьсот ', ' 700 '), (' восемьсот ', ' 800 '),
                     (' девятьсот ', ' 900 ')]

    stop_words = nltk.corpus.stopwords.words('russian') + nltk.corpus.stopwords.words('english')

    def __init__(self, n_features: int, vocabulary_file: str):
        self.ms = Mystem()
        self.exclude = set([x for x in set(self.stop_words + self.flt_chars) if len(x) > 3])
        self.max_lemm_word_length = 3
        self.n_features = n_features
        self.vocabulary_file = vocabulary_file
        self.v = Vectorizer(self.n_features, self.vocabulary_file)
        self.aug = Augmenter()

    def replace_garbage(self, x: str) -> str:
        """
        Очищает строку от знаков из списка-исключений и множественных пробелов
        TO-DO - разбить на два отдельных метода
        """
        for char in self.flt_chars:
            x = x.replace(char, ' ').lower()
        item = ' '.join(re.split(r'(\d+)', x))
        while item.count(2 * " ") > 0:
            item = item.replace(2 * " ", " ")
        return item.strip(' ').replace(' и ', ' ')

    def clean_text(self, x: str) -> str:
        """Превращает текст в строку слов начальной формы"""
        words = nltk.word_tokenize(x)
        words = [self.replace_garbage(word) for word in words if word not in self.exclude or len(word) > 3]

        lemmetized = []
        for word in words:
            lemmetized.append(' '.join(self.ms.lemmatize(word)).strip(' \n'))

        return self.replace_garbage(' '.join(lemmetized))

    def clean_df(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Получает датафрейм для верификации
        TO-DO - сделать проверку на наличие колонок
        """
        data["text"] = (
                data['description'] + ' '
                + data['category'].str.lower() + ' '
                + data['subcategory'].str.lower() + ' '
                + data['title'].str.lower()
        )

        print('Cleaning text field in dataframe...')

        for char in self.numbers_chars:
            data.loc[:, "text"] = data.loc[:, "text"].str.replace(*char).copy()

        data.loc[:, "text_cleaned"] = data.loc[:, "text"].apply(lambda x: self.clean_text(x)).copy()

        return data

    def create_vocabulary(self, train_data) -> None:
        print('Initializing vocabulary')
        vectorizer = self.v.init_vectorizer(load_vocab=False)
        text_data, y = train_data['text_cleaned'].to_list(), train_data.is_bad.to_list()
        vect_data = vectorizer.fit_transform(text_data)
        self.v.save_kbest_features_vocab(vectorizer, vect_data, y=y)
        return None

    def vectorize_to_nfeatures(self, data):
        print('Vectorizing text field in dataframe')
        text_data = data['text_cleaned'].to_list()
        vectorizer = self.v.init_vectorizer(load_vocab=True)
        return vectorizer.fit_transform(text_data)

    @staticmethod
    def filter_by_words_length(data: pd.DataFrame,
                               word_limit: int = 200, string_limit: int = 1000) -> pd.DataFrame:
        data["words_per_review"] = data["description"].str.split().apply(len)
        data = data[data["words_per_review"] < word_limit]
        data.loc[:, "description"] = data.loc[:, "description"].str[-string_limit:].copy()
        return data

    def vectorizing_pipeline(self, train_data: pd.DataFrame):
        """
            Returns:
                train_c: pd.DataFrame with cleaned text and target
                train_x: spacy sparce matrix, vectorized features w/o target
        """
        train_data_w = self.filter_by_words_length(train_data)
        train_data_c = self.clean_df(train_data_w)
        train_data_a = self.aug.augment(train_data_c)
        self.create_vocabulary(train_data_a)

        train_x = self.vectorize_to_nfeatures(train_data_a)
        return train_x, train_data_a
