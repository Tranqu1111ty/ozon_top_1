import numpy as np
import joblib
import json
import optuna
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
import pandas as pd
from catboost.text_processing import Tokenizer
from sklearn.metrics import accuracy_score, recall_score, mean_squared_error, f1_score, precision_score
import pymorphy2

dataset = pd.read_parquet(r"C:\Users\druzh\Project_python\ozon_top_1\Datasets\test_pairs_wo_target.parquet")
etl = pd.read_parquet(r"C:\Users\druzh\Project_python\ozon_top_1\Datasets\test_data.parquet")

features = (dataset.merge(etl.add_suffix('1'), on="variantid1").merge(etl.add_suffix('2'), on="variantid2"))

features["cat31"] = features["categories1"].apply(lambda x: json.loads(x)["3"])
cat3_counts = features["cat31"].value_counts().to_dict()

features["cat31_grouped"] = features["cat31"].apply(lambda x: x if cat3_counts[x] > 1000 else "rest")


print(features.head(2))

def extract_values(dict_string):
    if dict_string != None:
        dict_obj = json.loads(dict_string)
        result = []
        for value in dict_obj.values():
            if isinstance(value, list):
                result.append(' '.join(value))
            else:
                result.append(value)
        return ' '.join(result)
    return 'пусто'


features['characteristic_attributes_mapping1'] = features['characteristic_attributes_mapping1'].apply(extract_values)
features['characteristic_attributes_mapping2'] = features['characteristic_attributes_mapping2'].apply(extract_values)

feats = ['name1', 'name2', 'characteristic_attributes_mapping1', 'characteristic_attributes_mapping2',
         'cat31_grouped']

features = features[feats]

print(features.head(3))

tokenizer = Tokenizer(
    lowercasing=True,
    separator_type='BySense',
    token_types=['Word', 'Number']
)

stop_words = set(('для', 'это', 'в', 'на', 'где', 'из', 'с',))


def filter_stop_words(tokens):
    return list(filter(lambda x: x not in stop_words, tokens))


def lemmatize_tokens(tokens):
    m = pymorphy2.MorphAnalyzer()
    lemmas = [m.parse(token)[0].normal_form for token in tokens]
    return lemmas


def preprocess_data(X):
    X_preprocessed = X.copy()
    X_preprocessed['name1'] = X['name1'].apply(
        lambda x: ' '.join(filter_stop_words(tokenizer.tokenize(x))))
    X_preprocessed['name2'] = X['name2'].apply(
        lambda x: ' '.join(filter_stop_words(tokenizer.tokenize(x))))
    X_preprocessed['cat31_grouped'] = X['cat31_grouped'].apply(
        lambda x: ' '.join(filter_stop_words(tokenizer.tokenize(x))))
    X_preprocessed['characteristic_attributes_mapping1'] = X['characteristic_attributes_mapping1'].apply(
        lambda x: ' '.join(filter_stop_words(tokenizer.tokenize(x))))
    X_preprocessed['characteristic_attributes_mapping2'] = X['characteristic_attributes_mapping2'].apply(
        lambda x: ' '.join(filter_stop_words(tokenizer.tokenize(x))))

    return X_preprocessed


X_preprocessed_test = preprocess_data(features)


test_processed_pool = Pool(
    X_preprocessed_test,
    text_features=['name1', 'name2', 'characteristic_attributes_mapping1', 'characteristic_attributes_mapping2'],
    cat_features=['cat31_grouped']
)


best_model = CatBoostClassifier()

best_model.load_model("catboost_preproc_w_cats_n_chars.cbm")

prediction = best_model.predict_proba(test_processed_pool)

prediction = prediction[::, 1]

dataset['target'] = prediction

dataset.to_csv("last_sub.csv")
