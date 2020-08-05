import pandas as pd


#Per portare la ground_truth a livello di sorgenti e non di istanze.
df = pd.read_csv("camera_gt.csv")
schema = df.drop_duplicates(subset=['TARGET_ATTRIBUTE_ID','SOURCE_NAME'])
schema.to_csv("schema.csv",index=False)
