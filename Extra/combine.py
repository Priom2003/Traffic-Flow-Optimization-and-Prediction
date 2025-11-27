import pandas as pd

# Load the two CSV files
df1 = pd.read_csv("fetch_cleaned.csv")
df2 = pd.read_csv("traffic(use).csv")

# Combine (stack one below the other)
df_combined = pd.concat([df1, df2], ignore_index=True)

# Save to a new file
df_combined.to_csv("traffic_1(use).csv", index=False)
# import pandas as pd

# # === Input / Output paths ===
# file1 = "traffic_tweets.csv"
# file2 = "traffic_tweets_1.csv"
# file_out = "traffic_combined_aligned.csv"

# # === Load both files ===
# df1 = pd.read_csv(file1)
# df2 = pd.read_csv(file2)

# # === Align columns ===
# # Get union of columns from both datasets
# all_cols = list(set(df1.columns) | set(df2.columns))

# # Reindex both DataFrames to have same columns
# df1_aligned = df1.reindex(columns=all_cols)
# df2_aligned = df2.reindex(columns=all_cols)

# # === Combine ===
# df_combined = pd.concat([df1_aligned, df2_aligned], ignore_index=True)

# # === Save combined dataset ===
# df_combined.to_csv(file_out, index=False)

# print(f"Aligned combined dataset saved as {file_out} with {df_combined.shape[0]} rows and {df_combined.shape[1]} columns")
