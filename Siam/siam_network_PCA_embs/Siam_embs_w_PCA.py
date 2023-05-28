import numpy as np
import pandas as pd
import joblib
import os
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Subtract, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import optuna

dataset = pd.read_parquet(r"C:\Users\druzh\Project_python\ozon_top_1\Datasets\train_pairs.parquet")
etl = pd.read_parquet(r"C:\Users\druzh\Project_python\ozon_top_1\Datasets\train_data.parquet")

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
y = features['target'].values
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.1, random_state=42)

X1_train, X2_train = X_train[::, :len(X_train[0]) // 2], X_train[::, len(X_train[0]) // 2:]
X1_test, X2_test = X_test[::, :len(X_test[0]) // 2], X_test[::, len(X_test[0]) // 2:]

print("я тут все обработал, все хорошо, милорд!")
def create_siamese_network(input_shape, learning_rate, dropout_rate, num_neurons, num_layers):

    print("а мы тут все плюшками балуемся")
    input1 = Input(shape=input_shape)
    input2 = Input(shape=input_shape)

    dense_layer = Dense(num_neurons, activation='relu')

    x1 = dense_layer(input1)
    x1 = Dropout(dropout_rate)(x1)
    x2 = dense_layer(input2)
    x2 = Dropout(dropout_rate)(x2)

    for _ in range(num_layers):
        x1 = Dense(num_neurons, activation='relu')(x1)
        x1 = Dropout(dropout_rate)(x1)
        x2 = Dense(num_neurons, activation='relu')(x2)
        x2 = Dropout(dropout_rate)(x2)

    diff = Subtract()([x1, x2])

    output = Dense(1, activation='softmax')(diff)

    model = Model(inputs=[input1, input2], outputs=output)

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss='binary_crossentropy', optimizer=optimizer)

    return model


def objective(trial):
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-6, 2e-2)
    dropout_rate = trial.suggest_uniform('dropout_rate', 0.0, 0.5)
    num_neurons = trial.suggest_int('num_neurons', 512, 1024)
    num_layers = trial.suggest_int('num_layers', 1, 4)
    epochs = trial.suggest_int('epochs', 100, 300)

    input_shape = X1_train.shape[1:]
    siamese_model = create_siamese_network(input_shape, learning_rate, dropout_rate, num_neurons, num_layers)

    early_stopping = EarlyStopping(monitor='val_loss', patience=50)

    checkpoint_dir = r'C:\Users\druzh\Project_python\ozon_top_1\Siam'
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, 'best_model.h5')

    checkpoint_callback = ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, mode='max')

    siamese_model.fit([X1_train, X2_train], Y_train, epochs=epochs, batch_size=1, verbose=0,
                      validation_data=([X1_test, X2_test], Y_test),
                      callbacks=[early_stopping, checkpoint_callback])

    best_model = create_siamese_network(input_shape, learning_rate, dropout_rate, num_neurons, num_layers)
    best_model.load_weights(checkpoint_path)

    y_pred = best_model.predict([X1_test, X2_test])
    y_pred = np.round(y_pred).flatten()

    f1 = f1_score(Y_test, y_pred)
    return f1


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)

best_trial = study.best_trial
print('Best trial:')
print('  F1 Score: {:.6f}'.format(best_trial.value))
print('  Params:')
for key, value in best_trial.params.items():
    print('    {}: {}'.format(key, value))

model = load_model(r'C:\Users\druzh\Project_python\ozon_top_1\Siam\best_model.h5')
prediction = model.predict([X1_test, X2_test])
prediction = np.round(prediction).flatten()
f1 = f1_score(Y_test, prediction)
print(f1)
