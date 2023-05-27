import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

dataset = pd.read_parquet(r"E:\Py\AllBASE\OZON\Datasets\train_pairs.parquet")
etl = pd.read_parquet(r"E:\Py\AllBASE\OZON\Datasets\train_data.parquet")

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
              i].tolist() + compressed_embeddings_name_bert_642[i].tolist()
    X.append(row)

X = np.array(X)
y = features['target'].values


def train_and_evaluate_model(features):
    X_train, X_test, y_train, y_test = train_test_split(X[:, features], y, test_size=0.2)

    if X_train.shape[1] == 0:
        return 0.0

    model = xgb.XGBClassifier(n_estimators=300, max_depth=4)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    return accuracy


selected_features = []


for i in range(len(X[0])):

    accuracy = train_and_evaluate_model(selected_features)

    best_feature = None
    best_accuracy = accuracy

    for feature in range(len(X[0])):
        if feature not in selected_features:
            selected_features.append(feature)
            new_accuracy = train_and_evaluate_model(selected_features)
            selected_features.remove(feature)

            if new_accuracy > best_accuracy:
                best_feature = feature
                best_accuracy = new_accuracy

    if best_feature is not None:
        selected_features.append(best_feature)
    else:
        break


if len(selected_features) > 0:

    final_model = xgb.XGBClassifier(n_estimators=300, max_depth=4)
    final_model.fit(X[:, selected_features], y)
else:
    print("No features selected for training.")

joblib.dump(selected_features, "selected_features.joblib")
final_model.save_model("xgb_model.bin")
