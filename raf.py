import os
import json
import dizionario as d
import pre_tagging as p
import utils
import random
import math

with open("config.json") as c:
    config=json.load(c)


raf_path = "persistent_files/RAF_output_adapted_camera_june30.json"
with open(raf_path) as r:
    raf = json.load(r)

def create_training_raf():
    utils.new_file("train")
    utils.new_file("valid")
    utils.new_file("test")
    for attribute in config["ATTRIBUTES"]:
        name,atomic,non_atomic = get_output_from_name(attribute)
        random.shuffle(atomic)
        random.shuffle(non_atomic)
        name = name.replace(" ","_")
        train_a,train_n = atomic[:math.floor(len(atomic)*(config["train"]+config["valid"]))],non_atomic[:math.floor(len(non_atomic)*(config["train"]+config["valid"]))]
        test_a,test_n = atomic[math.floor(len(atomic)*(config["train"]+config["valid"])):],non_atomic[math.floor(len(non_atomic)*(config["train"]+config["valid"])):]
        #d.raf_dict(name,train_a)
        for s in train_a+train_n:
            source,attribute= s.split("__")[0],s.split("__")[1]
            _,attribute_name = attribute.split("/")
            directory_source = config["DIRECTORY_DATASET"]+source
            _,source_sentences=p.create_tagging_schema(directory_source,source,config["ATTRIBUTES"],True,attribute_name,name)
            random.shuffle(source_sentences)
            if source_sentences:
                utils.write("train",source_sentences[:math.floor(config["train"]*len(source_sentences))])
                utils.write("valid",source_sentences[math.floor(config["train"]*len(source_sentences)):])
        d.raf_dict(name,train_a+test_a)
        for s in test_a+test_n:
            source,attribute= s.split("__")
            _,attribute_name = attribute.split("/")
            directory_source = config["DIRECTORY_DATASET"]+source
            _,source_sentences=p.create_tagging_schema(directory_source,source,config["ATTRIBUTES"],True,attribute_name,name)
            random.shuffle(source_sentences)
            if source_sentences:
                utils.write("test",source_sentences)

def get_output_from_name(name):
    for attribute in raf:
        raf_name = attribute["__name"].split("__")[0].replace(" ","_")
        if raf_name==name:
            return (raf_name,attribute["atomic"],attribute["nonatomic"])
    return("Notfound","Notfound","Notfound")


def contain(frase,parola):
    for (token,tag) in frase:
        if token.lower()==parola:
            return True
    return False
