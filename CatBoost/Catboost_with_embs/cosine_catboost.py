import optuna
import numpy as np
import pandas as pd
import seaborn as sns
from catboost import CatBoostClassifier
from scipy.spatial.distance import cosine, euclidean
from sklearn.metrics import accuracy_score, recall_score, mean_squared_error, f1_score, precision_score
from sklearn.model_selection import train_test_split

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

main_pic_emb_1 = features['main_pic_embeddings_resnet_v11'].values
name_emb_1 = features['name_bert_641'].values
main_pic_emb_2 = features['main_pic_embeddings_resnet_v12'].values
name_emb_2 = features['name_bert_642'].values

main_pics_dist = [cosine(main_pic_emb_1[i][0], main_pic_emb_2[i][0]) for i in range(len(main_pic_emb_1))]
names_dist = [cosine(name_emb_1[i], name_emb_2[i]) for i in range(len(name_emb_1))]
print(main_pics_dist)
print(names_dist)

X = pd.DataFrame({'A': main_pics_dist,
                  'B': names_dist})

y = features['target'].values

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.1, random_state=42)


def objective(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 100, 1100),
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

study.optimize(objective, n_trials=100)

best_params = study.best_params
best_accuracy = study.best_value

print('Best parameters:', best_params)
print('Best accuracy:', best_accuracy)

model = CatBoostClassifier(**best_params)
model.fit(X_train, Y_train)
model.save_model("cosine_catboost_optuna.cbm")

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
