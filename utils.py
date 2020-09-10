import json
import nltk

with open("config.json") as c:
    config=json.load(c)

#Elimina gli elementi vuoti all'interno di un dizionario
def delete_empty(values_dict):
    for k in list(values_dict):
        try:
            if len(values_dict[k])<1:
                del values_dict[k]
        except:
            pass

def open_json(filename,source):
    if filename.endswith(".json"):
        with open(config["DIRECTORY_DATASET"]+source+"/"+filename) as json_file:
            return json.load(json_file)
    else:
        return "Not json"


def tokenizer(sentence):
    return nltk.word_tokenize(sentence)

def new_file(set):
    with open("training_files/"+set+"_set.txt","a") as f:
        f.truncate(0)
        f.write("-DOCSTART-	-X-	-X-	O\n\n")

def write(set,sentences):
    with open("training_files/"+set+"_set.txt","a") as f:
        for attr,sentence in sentences:
            for (token,tag) in sentence:
                f.write(token+"\t"+tag+"\n")
            f.write("\n")
        f.close()
