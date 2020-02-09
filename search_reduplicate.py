# %%
import pandas as pd
import re


# %%
df = pd.read_csv("model_five/pred.csv")
print(df)

# %%
pattern = r"([\u4e00-\u9fa5])\1"
test_str = "白云千载空悠悠"
print(re.search(pattern, test_str))

# %%
reduplicate_row_idx = []
for idx, row in df.iterrows():
    if re.search(pattern, row['src']):
        reduplicate_row_idx.append(idx)
print(reduplicate_row_idx)

# %%
reduplicate_row_df = df.iloc[reduplicate_row_idx]
r_df = reduplicate_row_df.drop(columns=['Unnamed: 0'])
r_df.to_csv("model_five/rpred.csv")


# %%
