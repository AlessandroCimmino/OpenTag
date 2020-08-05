import os
import pandas as pd
import json as j

df = pd.DataFrame(columns=['spec_source','spec_number','spec_id','spec_title'])
for dirpath, dirnames, files in os.walk('./2013_camera_dataset'):
    for f in files:
        if(os.path.splitext(f)[1]==".json"):
            with open(dirpath+"/"+f) as json_file:
                js = j.load(json_file)
                df = df.append({'spec_source':os.path.split(dirpath)[1],\
                            'spec_number':os.path.splitext(f)[0],\
                            'spec_id':(os.path.split(dirpath)[1]+"//"+os.path.splitext(f)[0]),\
                            "spec_title":js["<page title>"]},ignore_index=True)
df.to_csv("dataset.csv",index=False)
