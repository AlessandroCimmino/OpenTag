import json
import os
import pickle
import re
from itertools import combinations
import math
import random
import utils
import dizionario as d

with open("config.json") as c:
    config=json.load(c)

sources =config["SOURCES"]
source = ""

def tokens_to_tag(attribute_value,attributes_chosed):
    tokens_to_tag = []
    with open("persistent_files/"+ config["DIZIONARIO"], 'rb') as f:
        values_dict = pickle.load(f)
        d.choose_target_attribute(values_dict,attributes_chosed)
        utils.delete_empty(values_dict)
        for s in values_dict:
            for tag_attribute in values_dict[s]:
                values = [value for value in values_dict[s][tag_attribute] if attribute_value.find(value,0,len(attribute_value))>=0]
                if values:
                    for token in values:
                        tokens_to_tag.append((token,tag_attribute))
        return tokens_to_tag


def in_token_tag(token,tokens_to_tag):
    for (t,tag) in tokens_to_tag:
        if token==t:
            return True
    return False

def single_value_tag(attribute,tokens,tokens_tag,attributes_chosed,source,raf,raf_attribute,name):
    json_sentence = []
    last_tag="O"
    for token in tokens:
        predicate_names = [t_attr for (v,t_attr) in tokens_tag if v==token]
        if not raf:
            predicate_name = d.coeherent_attribute(attribute,source)
            if predicate_name in predicate_names and not re.match("[,:;()\\\/]",token):
                if last_tag=="O":
                    json_sentence.append((token,"B-"+predicate_name))
                    last_tag = "B"
                else:
                    json_sentence.append((token,"I-"+predicate_name))
                    last_tag = "I"
            else:
                json_sentence.append((token,"O"))
                last_tag = "O"
        else:
            if name in predicate_names and not re.match("[,:;()\\\/]",token):
                if last_tag=="O":
                    json_sentence.append((token,"B-"+name))
                    last_tag = "B"
                else:
                    json_sentence.append((token,"I-"+name))
                    last_tag = "I"
            else:
                json_sentence.append((token,"O"))
                last_tag = "O"
    return json_sentence

def best_tagging(attribute,value,tokens_tag,attributes_chosed,source,raf,raf_attribute,name):
    tokens = utils.tokenizer(value)
    l = 0
    json_sentence = []
    single_value = single_value_tag(attribute,tokens,tokens_tag,attributes_chosed,source,raf,raf_attribute,name)
    while l<len(tokens)-1:
        current = value.rsplit(' ',l)[0]
        predicate_names = [t_attr for (v,t_attr) in tokens_tag if v==current]
        if not raf:
            predicate_name = d.coeherent_attribute(attribute,source)
            if predicate_name in predicate_names:
                temp = 0
                last_tag = "O"
                for token in tokens:
                    if token in current and not re.match("[,:;()\\\/]",token):
                        if last_tag=="O":
                            json_sentence.append((token,"B-"+predicate_name))
                            last_tag = "B"
                        else:
                            json_sentence.append((token,"I-"+predicate_name))
                            last_tag = "I"
                    else:
                        json_sentence.append((token,"O"))
                        last_tag = "O"
                    temp+=1
        else:
            if name in predicate_names:
                temp = 0
                last_tag = "O"
                for token in tokens:
                    if token in current and not re.match("[,:;()\\\/]",token):
                        if last_tag=="O":
                            json_sentence.append((token,"B-"+name))
                            last_tag = "B"
                        else:
                            json_sentence.append((token,"I-"+name))
                            last_tag = "I"
                    else:
                        json_sentence.append((token,"O"))
                        last_tag = "O"
                    temp+=1
        if useful(json_sentence)>useful(single_value):
            return json_sentence
        l+=1
    return single_value


def tag_sentence(attribute,value,attributes_chosed,source,raf,raf_attribute,name):
    tokens_tag = tokens_to_tag(value,attributes_chosed)
    return best_tagging(attribute,value,tokens_tag,attributes_chosed,source,raf,raf_attribute,name)


def useful(sentence):
    count = 0
    for (token,tag) in sentence:
        if tag.find("B-",0)==0 or tag.find("I-",0)==0:
            count += 1
    return count

def create_tagging_schema(directory_source,source,attributes_chosed,raf=False,raf_attribute="",name=""):
    files = os.listdir(directory_source)
    source_sentences = []
    source_example_counts = dict()
    print("Tagging al the sentence of source: "+source+" ...")
    for filename in [file for file in files if file.endswith(".json")]:
        js = utils.open_json(filename,source)
        for attribute in js:
            if raf:
                if isinstance(js[attribute],str) and attribute!="<page title>" and attribute==raf_attribute:
                    sentence = []
                    for token in utils.tokenizer(attribute):
                        sentence.append((token,"O"))
                    sentence.append(("ENDNAME","O"))
                    sentence = sentence + tag_sentence(attribute,js[attribute],attributes_chosed,source,raf,raf_attribute,name)
                    sentence.append(("ENDVALUE","O"))
                    if useful(sentence)>0:
                        source_sentences.append((raf_attribute,sentence))
                        source_example_counts.setdefault(raf_attribute,0)
                        source_example_counts[raf_attribute]=source_example_counts[raf_attribute]+1
            else:
                if isinstance(js[attribute],str) and attribute!="<page title>" and [t for t in d.get_predicate_name(attribute,source,True) if t in attributes_chosed]:
                    sentence = []
                    for token in utils.tokenizer(attribute):
                        sentence.append((token,"O"))
                    sentence.append(("ENDNAME","O"))
                    sentence = sentence + tag_sentence(attribute,js[attribute],attributes_chosed,source)
                    sentence.append(("ENDVALUE","O"))
                    if useful(sentence)>0:
                        p_name = d.get_predicate_name(attribute,source,True)[0]
                        source_sentences.append((p_name,sentence))
                        source_example_counts.setdefault(p_name,0)
                        source_example_counts[p_name]=source_example_counts[p_name]+1
    return (source_example_counts,source_sentences)



def choosed_sources():
    attributes_chosed = []
    append=False
    for d in config["DATASETS"]:
        append=False
        print("Creating {0} set...".format(d))
        for s in sources[d]:
            source=s
            directory_source = config["DIRECTORY_DATASET"] + source
            specific = [(name,specific) for (name,specific) in config["SPECIFIC_ATTRIBUTES"] if name==source]
            if specific!=[]:
                attributes_chosed=specific[0][1]
            else:
                attributes_chosed=config["ATTRIBUTES"]
            _,source_sentences=create_tagging_schema(directory_source,source,attributes_chosed)
            with open("dataset/"+d+"_set.txt","a") as f:
                if not append:
                    f.truncate(0)
                    f.write("-DOCSTART-	-X-	-X-	O\n\n")
                for sentence in source_sentences:
                    for (token,tag) in sentence:
                        f.write(token+"\t"+tag+"\n")
                    f.write("\n")
                f.close()
            append=True


def tag_sources():
    all_sentences = []
    for s in [x[1] for x in os.walk(config["DIRECTORY_DATASET"])][0]:
        if s!="www.ebay.com" and s!="www.alibaba.com":
        #if s!="a":
            directory_source = config["DIRECTORY_DATASET"] + s
            specific = [(name,specific) for (name,specific) in config["SPECIFIC_ATTRIBUTES"] if name==source]
            if specific!=[]:
                attributes_chosed=specific[0][1]
            else:
                attributes_chosed=config["ATTRIBUTES"]
            example_count,source_sentences = create_tagging_schema(directory_source,s,attributes_chosed)
            all_sentences.append((s,example_count,source_sentences))
    return all_sentences

def not_disjoin_set(all_sentences):
    attribute_counts = dict()
    attribute_sentences = dict()
    for (s,counts,sentences) in all_sentences:
        for attribute,count in counts.items():
            attribute_counts.setdefault(attribute,0)
            attribute_counts[attribute]+=count
        for (attribute,sentence) in sentences:
            attribute_sentences.setdefault(attribute,[])
            attribute_sentences[attribute].append(sentence)
    train_sentences,valid_sentences,test_sentences=[],[],[]
    for key in attribute_sentences.keys():
        random.shuffle(attribute_sentences[key])
        train_len=math.floor(config["train"]*attribute_counts[key])
        valid_len=math.floor(config["valid"]*attribute_counts[key])
        test_len=math.floor(config["test"]*attribute_counts[key])
        train_sentences += attribute_sentences[key][:train_len]
        valid_sentences += attribute_sentences[key][train_len:train_len+valid_len]
        test_sentences += attribute_sentences[key][train_len+valid_len:]
    append=False
    with open("dataset/train_set.txt","a") as f:
        if not append:
            f.truncate(0)
            f.write("-DOCSTART-	-X-	-X-	O\n\n")
        for sentence in train_sentences:
            for (token,tag) in sentence:
                f.write(token+"\t"+tag+"\n")
            f.write("\n")
    f.close()
    with open("dataset/valid_set.txt","a") as f:
        if not append:
            f.truncate(0)
            f.write("-DOCSTART-	-X-	-X-	O\n\n")
        for sentence in valid_sentences:
            for (token,tag) in sentence:
                f.write(token+"\t"+tag+"\n")
            f.write("\n")
    f.close()
    with open("dataset/test_set.txt","a") as f:
        if not append:
            f.truncate(0)
            f.write("-DOCSTART-	-X-	-X-	O\n\n")
        for sentence in test_sentences:
            for (token,tag) in sentence:
                f.write(token+"\t"+tag+"\n")
            f.write("\n")
    f.close()

def disjoint_set(all_sentences):
    value_to_count=dict()
    sentences = [sen for s,c,sen in all_sentences]
    sentences = [item for sublist in sentences for item in sublist]
    for (attr,s) in sentences:
        for (token,tag) in s:
            if tag.startswith("B-",0) and not multi(s):
                value_to_count.setdefault(token,0)
                value_to_count[token]+=1
    value_to_count = {key:value for (key,value) in value_to_count.items() if value>=5}
    for set in config["DATASETS"]:
        with open("dataset/"+set+"_set.txt","a") as f:
            f.truncate(0)
            f.write("-DOCSTART-	-X-	-X-	O\n\n")
    temp = sentences
    i=0
    limit={"train":False,"valid":False,"test":False}
    tot={"train":0,"valid":0,"test":0}
    while not(limit["train"] and limit["valid"] and limit["test"]) and value_to_count.items():
        set=config["DATASETS"][i]
        if not limit[set]:
            value,examples=random.choice(list(value_to_count.items()))
            del value_to_count[value]
            value_sentences = get_sentences_from_value(value,temp)
            temp = [(attr,s) for (attr,s) in temp if s not in value_sentences]
            with open("dataset/"+set+"_set.txt","a") as f:
                for sentence in value_sentences:
                    for (token,tag) in sentence:
                        f.write(token+"\t"+tag+"\n")
                    f.write("\n")
            f.close()
            tot[set]+=len(value_sentences)
            if tot[set]>=config[set]*len(sentences):
                limit[set]=True
        if i+1>=len(config["DATASETS"]):
            i=0
        else:
            i+=1


def add_false_examples(set,target):
    sentences = []
    for s in [x[1] for x in os.walk(config["DIRECTORY_DATASET"])][0]:
        if s!="www.ebay.com" and s!="www.alibaba.com":
            directory_source = config["DIRECTORY_DATASET"] + s
            files = os.listdir(directory_source)
            print(directory_source+"...")
            for filename in [file for file in files if file.endswith(".json")]:
                js = utils.open_json(filename,s)
                sentence = []
                for attribute in js:
                    if d.get_predicate_name(attribute,s,False)[0]==target and isinstance(js[attribute],str) and js[attribute]!="Black":
                        for token in utils.tokenizer(attribute):
                            sentence.append((token,"O"))
                        sentence.append(("ENDNAME","O"))
                        for token in utils.tokenizer(js[attribute]):
                            sentence.append((token,"O"))
                        sentence.append(("ENDVALUE","O"))
                sentences.append(sentence)
    with open("dataset/"+set+"_set.txt","a") as f:
        for sentence in sentences:
            for (token,tag) in sentence:
                f.write(token+"\t"+tag+"\n")
            f.write("\n")
    f.close()


def create_all():
    all_sentences = []
    all_sentences=tag_sources()
    if not(config["DISJOINT"]):
        not_disjoin_set(all_sentences)
    else:
        disjoint_set(all_sentences)
        


def get_sentences_from_value(value,sentences):
    value_sentences = []
    for (attr,s) in sentences:
        for (token,tag) in s:
            if token==value and not multi(s):
                value_sentences.append(s)
    return value_sentences

def multi(sentence):
    one=False
    multi=False
    end=False
    for (token,tag) in sentence:
        if tag.startswith("B-") and not(one):
            one=True
        if one and tag.startswith("O"):
            end=True
        if tag.startswith("B-") and end:
            multi=True
    return multi
