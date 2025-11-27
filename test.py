import pandas as pd
df = pd.read_csv("phase3_structured_with_labels.csv")
print(df["traffic_label_numeric"].value_counts(normalize=True))