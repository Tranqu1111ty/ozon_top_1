import json
from functools import partial
from typing import List

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

features = features.head(60)

main_pic_embs_1 = features['main_pic_embeddings_resnet_v11'].values
main_pic_embs_2 = features['main_pic_embeddings_resnet_v12'].values
name_emb_1 = features['name_bert_641'].values
name_emb_2 = features['name_bert_642'].values

main_pic_embs_1 = [i[0] for i in main_pic_embs_1]
main_pic_embs_2 = [i[0] for i in main_pic_embs_2]

main_pic_embs_1 = np.array(main_pic_embs_1)
main_pic_embs_2 = np.array(main_pic_embs_2)

X = pd.DataFrame({'main_pic_1': main_pic_embs_1.tolist(),
                  'name_1': name_emb_1.tolist(),
                  'main_pic_2': main_pic_embs_2.tolist(),
                  'name_2': name_emb_2.tolist()})

Y = features['target']

print(X.head(2))

X['main_pic_1'] = X['main_pic_1'].astype('category').cat.codes
X['name_1'] = X['name_1'].astype('category').cat.codes
X['main_pic_2'] = X['main_pic_2'].astype('category').cat.codes
X['name_2'] = X['name_2'].astype('category').cat.codes

print(X.head(2))

# model = CatBoostClassifier(iterations=100, learning_rate=0.1, cat_features=[0, 1, 2, 3])
# model.fit(np.vstack((main_pic_embs_1, name_emb_1, main_pic_embs_2, name_emb_2)), Y)
