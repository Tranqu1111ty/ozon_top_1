import neptune
import matplotlib.pyplot as plt
from neptune.types import File
from ozon_top_1.Ozon_metric import pr_auc_macro
from sklearn.metrics import mean_squared_error
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score, precision_score, recall_score
from data_preproc import preproc
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve

run = neptune.init_run(
    project="tranquillity/Ozontop1",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI0MzgzMmQ3ZS1hZjViLTQyMzctOGNiNy00MDM0NDI3YWVlYmEifQ=="
)

run["params"] = {"Log_reg_iter": 1000, "Data %": 1, "Bag_of_words_columns": "Name, Cat, Color", "Emb_enable": 0}

X_train_final, y_train, X_test_final, X_test, y_test, y_test_met, cats = preproc(
    path0="C:/Users/druzh/Project_python/ozon_top_1/Datasets/")

model = LogisticRegression(max_iter=1000)
model.fit(X_train_final, y_train)

prediction = model.predict(X_test_final)

recall = recall_score(y_test, prediction)
precision = precision_score(y_test, prediction)
run["Accuracy"] = accuracy_score(y_test, prediction)
run["Precision"] = precision
run["Recall"] = recall
run["Mean Squared Error"] = mean_squared_error(y_test, prediction)
run["F1"] = f1_score(y_test, prediction)

X_test["scores"] = prediction
y_pred = X_test["scores"]
y_pred = y_pred.reset_index(drop=True)
run["Ozon metrics"] = pr_auc_macro(y_test_met, y_pred, cats, 0.75)

precision, recall, thrs = precision_recall_curve(y_test, y_pred)
fig, ax1 = plt.subplots(1)

ax1.plot(recall, precision)
ax1.axhline(y=0.75, color='grey', linestyle='-')

run["Prec_auc_curve_0.75"].upload(File.as_html(fig))

run.stop()
