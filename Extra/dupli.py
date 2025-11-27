# import pandas as pd

# # Load your CSV
# df = pd.read_csv("traffic_1(use).csv")

# # Count duplicates based on a specific column, e.g., 'Cleaned_text'
# duplicate_counts = df['raw_text_tweet'].value_counts()

# # Only keep rows where the count is more than 1 (i.e., duplicates)
# duplicates_only = duplicate_counts[duplicate_counts > 1]

# print(duplicates_only)
# import pandas as pd

# # Load CSV
# df = pd.read_csv("traffic_1(use).csv")

# # Remove duplicates based on a specific column, e.g., 'Cleaned_text'
# df_no_duplicates = df.drop_duplicates(subset=['raw_text_tweet'])

# # Get the probable count after duplicates removal
# count_after_removal = len(df_no_duplicates)

# print("Number of rows after removing duplicates:", count_after_removal)
import pandas as pd

# Load CSV
df = pd.read_csv("traffic_1(use).csv")

# Total number of rows before removing duplicates
total_count = len(df)

print("Total number of rows in the CSV:", total_count)

