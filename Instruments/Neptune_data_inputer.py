import neptune

run = neptune.init_run(
    project="tranquillity/Ozontop1",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI0MzgzMmQ3ZS1hZjViLTQyMzctOGNiNy00MDM0NDI3YWVlYmEifQ=="
)

run["params"] = {"iter_num": 3000, "learning_rate": 0.24, "Depth": 4,"Data %": 100, "Bag_of_words_columns": "Name, Cat, Color", "Emb_enable": 0}

run["Accuracy"] = 0.77539
run["Precision"] = 0.74749
run["Recall"] = 0.7400019
run["Mean Squared Error"] = 0.2246
run["F1"] = 0.74373
run["Ozon metrics"] = 0.0155

run.stop()