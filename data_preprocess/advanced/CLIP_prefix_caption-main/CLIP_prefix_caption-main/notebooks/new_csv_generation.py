import pandas as pd
from tqdm import tqdm

TOP800_PATH = r"D:\Hateful-Image-Project\data_preprocess\advanced\CLIP_prefix_caption-main\CLIP_prefix_caption-main\notebooks\top_800_info.csv"
ANALYSIS_PATH = r"D:\Hateful-Image-Project\data_preprocess\advanced\CLIP_prefix_caption-main\CLIP_prefix_caption-main\notebooks\analysis_results.csv"
OUTPUT_PATH = r"D:\Hateful-Image-Project\data_preprocess\advanced\CLIP_prefix_caption-main\CLIP_prefix_caption-main\notebooks\top_800_new_info.csv"

# 读取
df_top = pd.read_csv(TOP800_PATH)
df_analysis = pd.read_csv(ANALYSIS_PATH)

df_analysis["clean_image"] = df_analysis["image"].str.replace("_mask", "", regex=False)
df_analysis["clean_image"] = df_analysis["clean_image"].str.replace(".png", "", regex=False)

df_top["clean_image"] = df_top["img"].str.replace("img/", "", regex=False)
df_top["clean_image"] = df_top["clean_image"].str.replace(".png", "", regex=False)


rows = []
for _, row in tqdm(df_top.iterrows(), total=len(df_top), desc="Merging"):
    img_id = row["clean_image"]
    match = df_analysis[df_analysis["clean_image"] == img_id]
    if len(match) > 0:
        row["description"] = match.iloc[0]["description"]
        row["keywords"] = match.iloc[0]["keywords"]
    else:
        row["description"] = None
        row["keywords"] = None
    rows.append(row)

df_merged = pd.DataFrame(rows)

df_merged = df_merged.drop(columns=["clean_image"])

# 保存
df_merged.to_csv(OUTPUT_PATH, index=False)

print("Merged file saved to:", OUTPUT_PATH)
