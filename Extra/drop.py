import pandas as pd

df = pd.read_csv("fetch_tweets.csv")

df = df.drop_duplicates(subset=["raw_text_tweet"], keep="first").reset_index(drop=True)
print(f"Shape after removing duplicates: {df.shape}")
df.to_csv("fetch_cleaned.csv", index=False)