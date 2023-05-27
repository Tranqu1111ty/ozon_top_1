import json
import joblib
from functools import partial
from typing import List
import random
from sklearn.metrics import mean_squared_error
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from catboost import CatBoostClassifier, Pool
from catboost.utils import eval_metric
from scipy.spatial.distance import cosine, euclidean
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier


def get_pic_features(main_pic_embeddings_1,
                     main_pic_embeddings_2,
                     percentiles: List[int]):
    """Calculate distances percentiles for
    pairwise pic distances. Percentiles are useful
    when product has several pictures.
    """

    if main_pic_embeddings_1 is not None and main_pic_embeddings_2 is not None:
        main_pic_embeddings_1 = np.array([x for x in main_pic_embeddings_1])
        main_pic_embeddings_2 = np.array([x for x in main_pic_embeddings_2])

        dist_m = pairwise_distances(
            main_pic_embeddings_1, main_pic_embeddings_2
        )
    else:
        dist_m = np.array([[-1]])

    pair_features = []
    pair_features += np.percentile(dist_m, percentiles).tolist()

    return pair_features


def text_dense_distances(ozon_embedding, comp_embedding):
    """Calculate Euclidean and Cosine distances between
    ozon_embedding and comp_embedding.
    """
    pair_features = []
    if ozon_embedding is None or comp_embedding is None:
        pair_features = [-1, -1]
    elif len(ozon_embedding) == 0 or len(comp_embedding) == 0:
        pair_features = [-1, -1]
    else:
        pair_features.append(
            euclidean(ozon_embedding, comp_embedding)
        )
        cosine_value = cosine(ozon_embedding, comp_embedding)

        pair_features.append(cosine_value)

    return pair_features


def create_top_bag_of_words(sentences, top_words):
    vectorizer = CountVectorizer(max_features=top_words)
    bag_of_words = vectorizer.fit_transform(sentences)
    len_sentences = [len(sentence.split()) for sentence in sentences]

    return bag_of_words, len_sentences, vectorizer


def encode_sentences(sentences, vectorizer):
    encoded_sentences = vectorizer.transform(sentences)
    len_sentences = [len(sentence.split()) for sentence in sentences]
    feature_names = vectorizer.get_feature_names()

    decoded_sentences = []
    for encoded_sentence in encoded_sentences:
        decoded_sentence = [feature_names[i] for i in encoded_sentence.indices]
        decoded_sentences.append(decoded_sentence)

    return decoded_sentences, len_sentences


def train_logistic_regression(X, y, chunk_size, num_epochs=10):
    clf = SGDClassifier(loss='log')  # Используем логистическую регрессию
    num_samples = len(X)
    num_chunks = num_samples // chunk_size

    for epoch in range(num_epochs):
        print(epoch)
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = start_idx + chunk_size
            X_chunk = X[start_idx:end_idx]
            y_chunk = y[start_idx:end_idx]
            clf.partial_fit(X_chunk, y_chunk, classes=np.unique(y))

    return clf


dataset = pd.read_parquet(r"C:\Users\druzh\Project_python\ozon_top_1\Datasets/train_pairs.parquet")
etl = pd.read_parquet(r"C:\Users\druzh\Project_python\ozon_top_1\Datasets/train_data.parquet")

features = (dataset.merge(etl.add_suffix('1'), on="variantid1").merge(etl.add_suffix('2'), on="variantid2"))

get_pic_features_func = partial(get_pic_features, percentiles=[0, 25, 50])

features[["pic_dist_0_perc", "pic_dist_25_perc", "pic_dist_50_perc"]] = (
    features[["pic_embeddings_resnet_v11", "pic_embeddings_resnet_v12"]].apply(
        lambda x: pd.Series(get_pic_features_func(*x)), axis=1)
)

features[["main_pic_dist_0_perc", "main_pic_dist_25_perc", "main_pic_dist_50_perc"]] = (
    features[["main_pic_embeddings_resnet_v11", "main_pic_embeddings_resnet_v12"]].apply(
        lambda x: pd.Series(get_pic_features_func(*x)), axis=1
    )
)

features[["euclidean_name_bert_dist", "cosine_name_bert_dist"]] = (
    features[["name_bert_641", "name_bert_642"]].apply(
        lambda x: pd.Series(text_dense_distances(*x)), axis=1
    )
)

features["cat31"] = features["categories1"].apply(lambda x: json.loads(x)["3"])
cat3_counts = features["cat31"].value_counts().to_dict()

features["cat31_grouped"] = features["cat31"].apply(lambda x: x if cat3_counts[x] > 1000 else "rest")

features["cat32"] = features["categories2"].apply(lambda x: json.loads(x)["3"])
cat3_counts = features["cat32"].value_counts().to_dict()

features["cat32_grouped"] = features["cat32"].apply(lambda x: x if cat3_counts[x] > 1000 else "rest")

bag_of_words, len_sentences, names_bag_vectorizer = create_top_bag_of_words(
    np.hstack((features['name1'].values, features['name2'].values)), 5000)

bag_of_words = bag_of_words.toarray()

features["name1_bag"] = bag_of_words[:len(bag_of_words) // 2].tolist()

features["name2_bag"] = bag_of_words[len(bag_of_words) // 2:].tolist()

features["name1_len"] = len_sentences[:len(len_sentences) // 2]

features["name2_len"] = len_sentences[len(len_sentences) // 2:]

bag_of_words, len_sentences, cats_bag_vectorizer = create_top_bag_of_words(
    np.hstack((features['cat31'].values, features['cat32'].values)), 250)

bag_of_words = bag_of_words.toarray()

features = features.drop(
    ["name1", "categories1", "pic_embeddings_resnet_v11", "main_pic_embeddings_resnet_v11", "name_bert_641",
     "name2",
     "categories2", "pic_embeddings_resnet_v12", "main_pic_embeddings_resnet_v12", "name_bert_642",
     'characteristic_attributes_mapping1', 'characteristic_attributes_mapping2'], axis=1)

features = features.drop(['cat31', 'cat32'], axis=1)

features["cat31_bag"] = bag_of_words[:len(bag_of_words) // 2].tolist()

features["cat32_bag"] = bag_of_words[len(bag_of_words) // 2:].tolist()

features["cat31_len"] = len_sentences[:len(len_sentences) // 2]

features["cat32_len"] = len_sentences[len(len_sentences) // 2:]

color_1_prod = features['color_parsed1'].values
colors = []
for i in range(len(color_1_prod)):
    try:
        colors.append(color_1_prod[i][0])
    except:
        colors.append('None')
color_2_prod = features['color_parsed2'].values
for j in range(len(color_2_prod)):
    try:
        colors.append(color_2_prod[j][0])
    except:
        colors.append('None')

bag_of_words, len_sentences, colors_bag_vectorizer = create_top_bag_of_words(colors, 200)

bag_of_words = bag_of_words.toarray()

features["color1_bag"] = bag_of_words[:len(bag_of_words) // 2].tolist()

features["color2_bag"] = bag_of_words[len(bag_of_words) // 2:].tolist()

features = features.drop(["color_parsed1", "color_parsed2"], axis=1)

feats = ["name1_bag", "name1_len", "name2_bag", "name2_len", "cat31_bag", "cat32_bag", "cat31_len", "cat32_len",
         "color1_bag", 'color2_bag', "pic_dist_0_perc", "pic_dist_25_perc", "pic_dist_50_perc", "main_pic_dist_0_perc",
         "main_pic_dist_25_perc", "main_pic_dist_50_perc", "euclidean_name_bert_dist", "cosine_name_bert_dist"]

X_train, X_test = train_test_split(
    features[feats + ["target", "variantid1", "variantid2", "cat31_grouped"]],
    test_size=0.1, random_state=42, stratify=features[["target"]])

X_train, X_val = train_test_split(
    X_train[feats + ["target"]],
    test_size=0.1, random_state=42, stratify=X_train[["target"]])

cats = X_test["cat31_grouped"]
y_test_w_var = X_test[["target", "variantid1", "variantid2"]]
y_test = X_test[["target"]]

X_test = X_test.drop(["target", "variantid1", "variantid2", "cat31_grouped"], axis=1)

y_train = X_train["target"]
y_val = X_val["target"]

X_train = X_train.drop(["target"], axis=1)
X_val = X_val.drop(["target"], axis=1)

X_train_final = []

for i in range(len(X_train)):
    row = []
    row.extend(list(X_train[feats[0]].iloc[i]))
    row.extend(list(X_train[feats[2]].iloc[i]))
    row.extend(list(X_train[feats[4]].iloc[i]))
    row.extend(list(X_train[feats[5]].iloc[i]))
    row.extend(list(X_train[feats[8]].iloc[i]))
    row.extend(list(X_train[feats[9]].iloc[i]))
    row.extend([X_train[feats[b]].iloc[i] for b in [1, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]])
    X_train_final.append(row)

X_test_final = []

for i in range(len(X_test)):
    row = []
    row.extend(list(X_test[feats[0]].iloc[i]))
    row.extend(list(X_test[feats[2]].iloc[i]))
    row.extend(list(X_test[feats[4]].iloc[i]))
    row.extend(list(X_test[feats[5]].iloc[i]))
    row.extend(list(X_test[feats[8]].iloc[i]))
    row.extend(list(X_test[feats[9]].iloc[i]))
    row.extend([X_test[feats[b]].iloc[i] for b in [1, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]])
    X_test_final.append(row)

chunk_size = 15000
num_epochs = 200

model = train_logistic_regression(X_train_final, y_train, chunk_size, num_epochs)

joblib.dump(model, 'logistic_regression_w_embs_model_200_epochs.pkl')

prediction = model.predict(X_test_final)

mse = mean_squared_error(y_test.T, prediction)
print("Mean Squared Error:", mse)

f1 = f1_score(y_test.T, prediction)
print("f1:", f1)

accuracy = accuracy_score(y_test.T, prediction)
precision = precision_score(y_test.T, prediction)
recall = recall_score(y_test.T, prediction)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
