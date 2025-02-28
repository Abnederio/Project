import pandas as pd

df = pd.read_csv("coffee_dataset.csv")
print(df['Coffee Name'].unique())