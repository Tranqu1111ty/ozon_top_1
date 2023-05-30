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

dataset = pd.read_parquet(r"C:\Users\druzh\Project_python\ozon_top_1\Datasets\train_pairs.parquet")
etl = pd.read_parquet(r"C:\Users\druzh\Project_python\ozon_top_1\Datasets\train_data.parquet")

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

feats = ['target', 'name1', 'name2', 'characteristic_attributes_mapping1', 'characteristic_attributes_mapping2',
         'cat31_grouped']

features = features[feats]

print(features.head(3))

train_df, test_df = train_test_split(features, train_size=0.8, random_state=25)

y_train, X_train = train_df['target'], train_df.drop(['target'], axis=1)
y_test, X_test = test_df['target'], test_df.drop(['target'], axis=1)

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


X_preprocessed_train = preprocess_data(X_train)
X_preprocessed_test = preprocess_data(X_test)

train_processed_pool = Pool(
    X_preprocessed_train, y_train,
    text_features=['name1', 'name2', 'characteristic_attributes_mapping1', 'characteristic_attributes_mapping2'],
    cat_features=['cat31_grouped']
)

test_processed_pool = Pool(
    X_preprocessed_test, y_test,
    text_features=['name1', 'name2', 'characteristic_attributes_mapping1', 'characteristic_attributes_mapping2'],
    cat_features=['cat31_grouped']
)

print(X_preprocessed_test.head(2))


def objective(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 300, 2000),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-1),
        'depth': trial.suggest_int('depth', 2, 6),
        'l2_leaf_reg': trial.suggest_loguniform('l2_leaf_reg', 1e-2, 10),
        'border_count': trial.suggest_int('border_count', 32, 255),
    }

    model = CatBoostClassifier(
        eval_metric='AUC',
        **params
    )

    return model.fit(
        train_processed_pool,
        eval_set=test_processed_pool,
        verbose=50,
        early_stopping_rounds=200
    ).best_score_['validation']['AUC']


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=110)

best_model = CatBoostClassifier(**study.best_params)
best_model.fit(train_processed_pool,
               eval_set=test_processed_pool,
               verbose=50,
               early_stopping_rounds=100)

best_model.save_model("catboost_preproc_w_cats_n_chars.cbm")

prediction = best_model.predict(test_processed_pool)

f1 = f1_score(y_test, prediction)
print("f1:", f1)

accuracy = accuracy_score(y_test, prediction)
precision = precision_score(y_test, prediction)
recall = recall_score(y_test, prediction)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
iter = 999
learning = 0.09089543434137001
depth = 5
lg_leaf_reg = 0.9192641887461273
border_count = 221
