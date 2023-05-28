import optuna
from catboost import CatBoostClassifier
from keras.models import Sequential
from keras.layers import Dense
import keras
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
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

feats = ["main_pic_embeddings_resnet_v11", "name_bert_641", "main_pic_embeddings_resnet_v12", "name_bert_642"]

X_train = features[feats]
y_train = features['target']

X_train_final = []

for i in range(len(X_train)):
    row = []
    row.extend(list(X_train[feats[0]].iloc[i])[0])
    row.extend(list(X_train[feats[1]].iloc[i]))
    row.extend(list(X_train[feats[2]].iloc[i])[0])
    row.extend(list(X_train[feats[3]].iloc[i]))
    X_train_final.append(row)

X_train_final = np.array(X_train_final)
y_train = np.array(y_train)

X_reshaped = np.expand_dims(X_train_final, axis=2)

print(X_reshaped)
model = keras.models.Sequential([
    keras.layers.Conv1D(filters=64, kernel_size=2,
                        strides=4, padding="causal",
                        activation="relu",
                        input_shape=[384, 1]),
    keras.layers.Conv1D(filters=32, kernel_size=2,
                        strides=4, padding="causal",
                        activation="relu"),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train_final, y_train, epochs=150)

model.save('fit_forward')
pred = model.predict(X_train_final)

embeddings = []
for i in features['main_pic_embeddings_resnet_v11'].values:
    embeddings.append(i[0])

embeddings = np.array(embeddings)

pca = PCA(n_components=30)

compressed_embeddings_main_pic = pca.fit_transform(embeddings)

name_bert_641_embedding = []
for i in features['name_bert_641'].values:
    name_bert_641_embedding.append(i)

name_bert_641_embedding_final = np.array(name_bert_641_embedding)

pca = PCA(n_components=30)

compressed_embeddings_name_bert_641 = pca.fit_transform(name_bert_641_embedding_final)
embeddings_main_2 = []
for i in features['main_pic_embeddings_resnet_v12'].values:
    embeddings_main_2.append(i[0])

embeddings_main_2_final = np.array(embeddings_main_2)

pca = PCA(n_components=30)

compressed_embeddings_main_pic_2 = pca.fit_transform(embeddings_main_2_final)
name_bert_642_embedding = []
for i in features['name_bert_642'].values:
    name_bert_642_embedding.append(i)

name_bert_642_embedding_final = np.array(name_bert_642_embedding)

pca = PCA(n_components=30)

compressed_embeddings_name_bert_642 = pca.fit_transform(name_bert_642_embedding_final)

X = []
for i in range(len(compressed_embeddings_main_pic)):
    row = compressed_embeddings_main_pic[i].tolist() + compressed_embeddings_name_bert_641[i].tolist() + \
          compressed_embeddings_main_pic_2[
              i].tolist() + compressed_embeddings_name_bert_642[i].tolist() + pred[i].tolist()
    X.append(row)

X = np.array(X)

print(len(X[0]))


X_train, X_test, Y_train, Y_test = train_test_split(X, y_train, test_size=0.1, random_state=42)


def objective(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 100, 1000),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.000001, 0.0002),
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


model.save_model("fit_forward_with_catboost.cbm")

