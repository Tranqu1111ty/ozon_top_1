import numpy as np
import pandas as pd
from tensorflow.keras.models import Model, load_model, load_weights
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
import optuna

dataset = pd.read_parquet(r"C:\Users\druzh\Project_python\ozon_top_1\Datasets\test_pairs_wo_target.parquet")
etl = pd.read_parquet(r"C:\Users\druzh\Project_python\ozon_top_1\Datasets\test_data.parquet")

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
          compressed_embeddings_main_pic_2[i].tolist() + compressed_embeddings_name_bert_642[i].tolist()
    X.append(row)

X = np.array(X)

X1_train, X2_train = X[::, :len(X[0]) // 2], X[::, len(X[0]) // 2:]

print("я тут все обработал, все хорошо, милорд!")

model = load_weights('best_model.h5')
prediction = model.predict([X1_train, X2_train])

print(prediction)
