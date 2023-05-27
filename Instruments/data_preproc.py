import json
from functools import partial
from typing import List
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split


# Мешок слов-крафтер
def create_bag_of_words(sentences):
    vectorizer = CountVectorizer()
    bag_of_words = vectorizer.fit_transform(sentences)
    len_sentences = []
    for sentence in sentences:
        len_sentences.append(len(sentence.split()))

    return bag_of_words.toarray(), len_sentences


def preproc(path0):
    # Читаем исходные данные
    dataset = pd.read_parquet(path0 + "train_pairs_w_target.parquet")
    etl = pd.read_parquet(path0 + "train_data.parquet")

    # dataset.head(2)
    # dataset["target"].value_counts()
    # etl.head(2)
    # print(etl.shape, dataset.shape)

    # Делаем Join подстановку данных
    features0 = (dataset.merge(etl.add_suffix('1'), on="variantid1").merge(etl.add_suffix('2'), on="variantid2"))

    # features0.head(2)

    # Берем только 1% данных (Мало памяти)
    features = features0[:3000]

    # Раскрываем категории для каждого товара
    features["cat31"] = features["categories1"].apply(lambda x: json.loads(x)["3"])
    cat3_counts = features["cat31"].value_counts().to_dict()

    features["cat31_grouped"] = features["cat31"].apply(lambda x: x if cat3_counts[x] > 1000 else "rest")

    features["cat32"] = features["categories2"].apply(lambda x: json.loads(x)["3"])
    cat3_counts = features["cat32"].value_counts().to_dict()

    features["cat32_grouped"] = features["cat32"].apply(lambda x: x if cat3_counts[x] > 1000 else "rest")

    # Создаем мешок слов по именам (Имена всех товаров)
    bag_of_words, len_sentences = create_bag_of_words(np.hstack((features['name1'].values, features['name2'].values)))
    features = features.drop(
        ["name1", "categories1", "pic_embeddings_resnet_v11", "main_pic_embeddings_resnet_v11", "name_bert_641",
         "name2",
         "categories2", "pic_embeddings_resnet_v12", "main_pic_embeddings_resnet_v12", "name_bert_642"], axis=1)
    # print(bag_of_words.shape)

    bag_of_words_1, bag_of_words_2 = bag_of_words[:len(bag_of_words) // 2], bag_of_words[len(bag_of_words) // 2:]
    len_sentences_1, len_sentences_2 = len_sentences[:len(len_sentences) // 2], len_sentences[len(len_sentences) // 2:]

    # Добавляем в df
    features["name1_bag"] = bag_of_words_1.tolist()
    features["name2_bag"] = bag_of_words_2.tolist()
    features["name1_len"] = len_sentences_1
    features["name2_len"] = len_sentences_2

    bag_of_words, len_sentences = create_bag_of_words(np.hstack((features['cat31'].values, features['cat32'].values)))
    features = features.drop(['cat31', 'cat32'], axis=1)
    # print(bag_of_words.shape)

    bag_of_words_1, bag_of_words_2 = bag_of_words[:len(bag_of_words) // 2], bag_of_words[len(bag_of_words) // 2:]
    len_sentences_1, len_sentences_2 = len_sentences[:len(len_sentences) // 2], len_sentences[len(len_sentences) // 2:]

    # Добавляем в df
    features["cat31_bag"] = bag_of_words_1.tolist()
    features["cat32_bag"] = bag_of_words_2.tolist()
    features["cat31_len"] = len_sentences_1
    features["cat32_len"] = len_sentences_2

    # features.head(2)

    # Формируем массив с названиями цветов
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

    # Мешок слов по цветам
    bag_of_words, _ = create_bag_of_words(colors)
    features = features.drop(['color_parsed1', 'color_parsed2'], axis=1)
    # print(bag_of_words.shape)

    bag_of_words_1, bag_of_words_2 = bag_of_words[:len(bag_of_words) // 2], bag_of_words[len(bag_of_words) // 2:]

    # Добавляем в df без длины (т.к. нет смысла)
    features["color1_bag"] = bag_of_words_1.tolist()
    features["color2_bag"] = bag_of_words_2.tolist()

    # features.head(2)

    # Формируем конечный df
    feats = ["name1_bag", "name1_len", "name2_bag", "name2_len", "cat31_bag", "cat32_bag", "cat31_len", "cat32_len",
             "color1_bag", 'color2_bag']
    X_train, X_test = train_test_split(
        features[feats + ["target", "variantid1", "variantid2", "cat31_grouped"]],
        test_size=0.1, random_state=42, stratify=features[["target"]])

    X_train, X_val = train_test_split(
        X_train[feats + ["target"]],
        test_size=0.1, random_state=42, stratify=X_train[["target"]])

    cats = X_test["cat31_grouped"]

    y_test = X_test[["target"]]
    X_test = X_test.drop(["target", "variantid1", "variantid2", "cat31_grouped"], axis=1)

    y_train = X_train["target"]
    X_train = X_train.drop(["target"], axis=1)

    # Из df в двумерный массив без лишиних разделителей
    X_train_final = []

    for i in range(len(X_train)):
        row = []
        row.extend(list(X_train[feats[0]].iloc[i]))
        row.extend(list(X_train[feats[2]].iloc[i]))
        row.extend(list(X_train[feats[4]].iloc[i]))
        row.extend(list(X_train[feats[5]].iloc[i]))
        row.extend(list(X_train[feats[8]].iloc[i]))
        row.extend(list(X_train[feats[9]].iloc[i]))
        row.extend([X_train[feats[b]].iloc[i] for b in [1, 3, 6, 7]])
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
        row.extend([X_test[feats[b]].iloc[i] for b in [1, 3, 6, 7]])
        X_test_final.append(row)

    y_test_met = y_test.reset_index(drop=True)
    cats = cats.reset_index(drop=True)
    y_test_met = y_test_met.T

    # print(len(X_train_final))
    # print(y_train)
    # print(len(X_test_final))
    # print(y_test)
    return X_train_final, y_train, X_test_final, X_test, y_test, y_test_met, cats

# path0 = "C:/Users/druzh/Project_python/ozon_top_1/Datasets/"
#
# X_train_final, y_train, X_test_final, y_test, y_test_w_var, cats = preproc(path0)
# print(len(X_train_final))
