import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
import pandas as pd
from catboost.text_processing import Tokenizer, Dictionary
import nltk
import os
from pymystem3 import Mystem
import pymorphy2

# dataset = pd.read_parquet(r"C:\Users\druzh\Project_python\ozon_top_1\Datasets\train_pairs.parquet")
# etl = pd.read_parquet(r"C:\Users\druzh\Project_python\ozon_top_1\Datasets\train_data.parquet")
#
# features = (dataset.merge(etl.add_suffix('1'), on="variantid1").merge(etl.add_suffix('2'), on="variantid2"))
#
# feats = ['target', 'name1', 'name2']
#
# features = features[feats]
#
# text = features['name1'].values[56734]
token = ['картридж', 'лазерный', 'комус', '729', 'пурпурный', '7018']
token = ['Бегущая', 'силиконовая', 'GSMIN']
def lemmatize_tokens(tokens):
    m = pymorphy2.MorphAnalyzer()
    lemmas = [m.parse(token)[0].normal_form for token in tokens]  # мы берем [0] индекс, потому что lemmatize возвращает список
    return lemmas

print(lemmatize_tokens(token))

