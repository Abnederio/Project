import pandas as pd

df = pd.read_csv("Balanced_CoffeeDS_500.csv")
print(df["Weather"].unique())