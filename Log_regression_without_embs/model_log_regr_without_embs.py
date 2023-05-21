from sklearn.metrics import mean_squared_error
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score, precision_score, recall_score
from data_preproc import preproc
from sklearn.linear_model import LogisticRegression

X_train_final, y_train, X_test_final, y_test, y_test_w_var, cats = preproc(
    path0="C:/Users/druzh/Project_python/ozon_top_1/Datasets/")

model = LogisticRegression(max_iter=1000)
model.fit(X_train_final, y_train)

prediction = model.predict(X_test_final)

mse = mean_squared_error(y_test, prediction)
print("Mean Squared Error:", mse)

f1 = f1_score(y_test, prediction)
print("f1:", f1)

accuracy = accuracy_score(y_test, prediction)
precision = precision_score(y_test, prediction)
recall = recall_score(y_test, prediction)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
