import optuna
import numpy as np
import pandas as pd

from sklearn.manifold import TSNE
from catboost import CatBoostClassifier
from sklearn.metrics import mean_squared_error
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score
from sklearn.metrics import precision_score


dataset = pd.read_parquet(r"C:\Users\druzh\Project_python\ozon_top_1\Datasets/train_pairs.parquet")
etl = pd.read_parquet(r"C:\Users\druzh\Project_python\ozon_top_1\Datasets/train_data.parquet")

features = (
    dataset
    .merge(
        etl
        .add_suffix('1'),
        on="variantid1"
    )
    .merge(
        etl
        .add_suffix('2'),
        on="variantid2"
    )
)

features['main_pic_embeddings_resnet_v11'].values[0][0]

embeddings = []
for i in features['main_pic_embeddings_resnet_v11'].values:
    embeddings.append(i[0])

embeddings = np.array(embeddings)

tsne_model = TSNE(n_components=3, perplexity=30, learning_rate=200)
compressed_embeddings_main_pic = tsne_model.fit_transform(embeddings)

name_bert_641_embedding = np.array(features['name_bert_641'].values)

name_bert_641_embedding = []
for i in features['name_bert_641'].values:
    name_bert_641_embedding.append(i)

name_bert_641_embedding_final = np.array(name_bert_641_embedding)

tsne_model = TSNE(n_components=3, perplexity=30, learning_rate=200)
compressed_embeddings_name_bert_641 = tsne_model.fit_transform(name_bert_641_embedding_final)

embeddings_main_2 = []
for i in features['main_pic_embeddings_resnet_v12'].values:
    embeddings_main_2.append(i[0])

embeddings_main_2_final = np.array(embeddings_main_2)

tsne_model = TSNE(n_components=3, perplexity=30, learning_rate=200)

compressed_embeddings_main_pic_2 = tsne_model.fit_transform(embeddings_main_2_final)

name_bert_642_embedding = []
for i in features['name_bert_642'].values:
    name_bert_642_embedding.append(i)

name_bert_642_embedding_final = np.array(name_bert_642_embedding)

tsne_model = TSNE(n_components=3, perplexity=30, learning_rate=200)
compressed_embeddings_name_bert_642 = tsne_model.fit_transform(name_bert_642_embedding_final)

X = []
for i in range(len(compressed_embeddings_main_pic)):
    row = compressed_embeddings_main_pic[i].tolist() + compressed_embeddings_name_bert_641[i].tolist() + \
          compressed_embeddings_main_pic_2[
              i].tolist() + compressed_embeddings_name_bert_642[i].tolist()
    X.append(row)

X = np.array(X)
y = features['target'].values

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.1, random_state=42)


def objective(trial):
    global X_test
    global Y_test
    global Y_train
    global X_train
    params = {
        'iterations': trial.suggest_int('iterations', 100, 1000),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.5),
        'depth': trial.suggest_int('depth', 3, 10),
        'l2_leaf_reg': trial.suggest_loguniform('l2_leaf_reg', 0.1, 10.0),
        'border_count': trial.suggest_int('border_count', 5, 255),
        'random_seed': 42,
        'eval_metric': 'Accuracy',
        'loss_function': 'Logloss',
        'verbose': False
    }

    model = CatBoostClassifier(**params)
    model.fit(X_train, Y_train, eval_set=(X_test, Y_test), early_stopping_rounds=20, verbose=False)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(Y_test, y_pred)
    return accuracy


study = optuna.create_study(direction='maximize')

study.optimize(objective, n_trials=50)

best_params = study.best_params
best_accuracy = study.best_value

print('Best parameters:', best_params)
print('Best accuracy:', best_accuracy)

model = CatBoostClassifier(**best_params)
model.fit(X_train, Y_train)

prediction = model.predict(X_test)

mse = mean_squared_error(Y_test, prediction)
print("Mean Squared Error:", mse)

f1 = f1_score(Y_test, prediction)
print("f1:", f1)

accuracy = accuracy_score(Y_test, prediction)
precision = precision_score(Y_test, prediction)
recall = recall_score(Y_test, prediction)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)

model.save_model("catboost_with_embs_only_TSNE.cbm")
