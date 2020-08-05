import os
import pickle
import utils
import json
import pandas
import re

with open("config.json") as c:
    config=json.load(c)

camera_gt = pandas.read_csv("persistent_files/schema.csv",index_col=0)

#Crea il dizionario per una sorgente che contiene i tutti i valori
#di tutti i gli attributi presenti all'interno dei json
def source_dictionary(source):
    d = dict()
    print("Creating values dictionary for : "+ source)
    for filename in os.listdir(config["DIRECTORY_DATASET"]+source):
        js = utils.open_json(filename,source)
        if js!="Not json":
            for attribute in js:
                predicate_name = get_predicate_name(attribute,source,False)[0]
                if isinstance(js[attribute], str) and predicate_name!="Not found":
                    for value in re.split('[\(\);,]',js[attribute]):
                        if value.lstrip() and len(value.lstrip())>1:
                            d.setdefault(predicate_name,set()).add(value.lstrip())
    return d

def raf_dict(ta,atomic):
    values_dict = dict()
    for s in atomic:
        source,raf_attribute= s.split("__")[0],(s.split("__")[1]).split("/")[1]
        print("Creating values dictionary for {0} and attribute {1}".format(source,raf_attribute))
        if source in values_dict.keys():
            source_dic = values_dict[source]
        else:
            values_dict.setdefault(source,dict())
            source_dic = dict()
        for filename in os.listdir(config["DIRECTORY_DATASET"]+source):
            js = utils.open_json(filename,source)
            if js!="Not json":
                for attribute in js:
                    if attribute==raf_attribute:
                        if isinstance(js[attribute],list):
                            for val in js[attribute]:
                                for value in re.split('(?<!\d)[.](?!\d)|[\(\)\/;,]',val):
                                    if value.lstrip() and len(value.lstrip())>1 and (value.lower().find("nikon",0,len(value))==-1):
                                        source_dic.setdefault(ta,set()).add(value.lstrip())
                        if isinstance(js[attribute],str):
                            for value in re.split('(?<!\d)[.](?!\d)|[\(\)\/;,]',js[attribute]):
                                if value.lstrip() and len(value.lstrip())>1 and (value.lower().find("nikon",0,len(value))==-1):
                                    source_dic.setdefault(ta,set()).add(value.lstrip())
        values_dict[source]=source_dic
    f = open("persistent_files/dizionario.pkl","wb")
    pickle.dump(values_dict,f)
    f.close()






#Crea il dizionario per tutte le sorgenti
def create_values_dictionary():
    values_dict = dict()
    for dir in [x[1] for x in os.walk(config["DIRECTORY_DATASET"])][0]:
        source_dic = source_dictionary(dir)
        if source_dic:
            values_dict.setdefault(dir,dict())
            values_dict[dir] = source_dic
    f = open("persistent_files/dizionario.pkl","wb")
    pickle.dump(values_dict,f)
    f.close()



#Sceglie le entry del dizionario per i target_attribute che si sono scelti
#per l'iterazione di opentag.
#Definiti all'interno di config["ATTRIBUTES"]
def choose_target_attribute(values_dict,attributes_chosed):
    for source in values_dict:
        for key in list(values_dict[source]):
            if not key in attributes_chosed:
                del values_dict[source][key]


def coeherent_attribute(attribute_name,source):
    predicate_name = get_predicate_name(attribute_name,source,True)
    if predicate_name:
        return predicate_name[0]
    else:
        return ""

def to_json(dizionario):
    with open("persistent_files/"+ dizionario, 'rb') as f:
        values_dict = pickle.load(f)
    dizionario = {}
    for source in values_dict:
        dizionario.setdefault(source,dict())
        for target_attribute in values_dict[source]:
            dizionario[source].setdefault(target_attribute,list())
            for value in values_dict[source][target_attribute]:
                dizionario[source][target_attribute].append(value)
    with open('persistent_files/dizionario.json', 'w') as fp:
        json.dump(dizionario, fp)

def to_pkl():
    with open('persistent_files/dizionario.json', 'r') as fp:
        data=json.load(fp)
        f = open("persistent_files/dizionario.pkl","wb")
        pickle.dump(data,f)
        f.close()


def del_elem_from_dict(value,attribute):
    with open("persistent_files/dizionario_pulito.pkl",'rb') as f:
        d = pickle.load(f)
    for s in d:
        if attribute in d[s]:
            temp = list(d[s][attribute])
            for v in temp:
                if value in v.lower():
                    print(v)
                    d[s][attribute].remove(v)
                    print(d[s][attribute])
            print("\n\n")
    with open("persistent_files/dizionario_modificato.pkl","wb") as m:
        pickle.dump(d,m)

def get_predicate_name(attribute,source,multi_predicate=False):
    try:
        predicate_name =camera_gt.loc[(camera_gt["SOURCE_NAME"]==source)
                            & (camera_gt["ATTRIBUTE_NAME"]==attribute),
                            "predicate_name"]
        if multi_predicate:
            if predicate_name.tolist():
                return predicate_name.values.tolist()
            else:
                reutrn ["Not found"]
        else:
            if len(predicate_name.tolist())==1:
                return predicate_name.values.tolist()
            else:
                return ["Not found"]
    except:
        return ["Not found"]

#to_json("dizionario.pkl")
to_pkl()
