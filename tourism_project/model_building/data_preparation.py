
import pandas as pd
from sklearn.model_selection import train_test_split

dataset_url = "https://huggingface.co/datasets/SriSair/tourism-dataset/resolve/main/tourism.csv"

df = pd.read_csv(dataset_url)

df.drop(
    columns=["Unnamed: 0", "CustomerID"],
    inplace=True,
    errors="ignore"
)

X = df.drop("ProdTaken", axis=1)
y = df["ProdTaken"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

train_df = pd.concat([X_train, y_train], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)

train_df.to_csv("tourism_project/data/train.csv", index=False)
test_df.to_csv("tourism_project/data/test.csv", index=False)

print("Data preparation completed.")
