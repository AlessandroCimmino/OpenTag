import os
import codecs
import numpy
import keras
import pickle
from keras_wc_embd import get_dicts_generator, get_batch_input
from models.model_bert import build_model
import re
import pre_tagging as p
import bert.bert as bert
import bert.start_server_bert as start_bert
import itertools
from operator import methodcaller
import json
import utils
import raf

with open("config.json") as c:
    config=json.load(c)


if config["CREATE_DICTIONARY"]:
    p.create_values_dictionary()
if config["CREATE_TRAINING"]:
    if config["ALL_SOURCES"]:
        p.create_all()
    else:
        p.choosed_sources()
if config["RAF"]:
    raf.create_training_raf()


numpy.set_printoptions(threshold=50)

DATA_TRAIN_PATH = os.path.join(config["DATA_ROOT"], 'train_set.txt')
DATA_VALID_PATH = os.path.join(config["DATA_ROOT"], 'valid_set.txt')
DATA_TEST_PATH = os.path.join(config["DATA_ROOT"], config["TEST_SET"])



def creazione_set_tag():
    tags = set()
    datasets = [config["TEST_SET"],"valid_set.txt","train_set.txt"]
    for txt in datasets:
        with open("dataset/"+txt) as f:
            for line in f:
                for token in line.split("\t"):
                    if re.match("(^B-.*)|(^I-.*)|(^O$)",token):
                        tags.add(token.rstrip())
    return tags


def create_tag ():
    TAGS = {'O':0}
    i = 1
    temp = set()
    tags = creazione_set_tag()
    for elem in tags:
        TAGS[elem]= i
        i = i +1
    print(TAGS)
    return TAGS

TAGS = config["TAGS"]


def load_data(path):
    sentences, taggings = [], []
    with codecs.open(path, 'r', 'utf8') as reader:
        for line in reader:
            line = line.strip()
            if not line:
                if not sentences or len(sentences[-1]) > 0:
                    sentences.append([])
                    taggings.append([])
                continue
            parts = line.split()
            if parts[0] != '-DOCSTART-':
                sentences[-1].append(parts[0])
                taggings[-1].append(TAGS[parts[-1]])
    if not sentences[-1]:
        sentences.pop()
        taggings.pop()
    return sentences, taggings


print('Loading...')
train_sentences, train_taggings = load_data(DATA_TRAIN_PATH)
valid_sentences, valid_taggings = load_data(DATA_VALID_PATH)
test_sentences, test_taggings = load_data(DATA_TEST_PATH)
test_steps = (len(test_sentences) + config["BATCH_SIZE"] - 1) // config["BATCH_SIZE"]




train_s=[]
train_t=[]
for sentence,sentence_t in zip(train_sentences,train_taggings):
    s = ""
    for word in sentence[0:149]:
        s = s +" "+ word
    train_t.append(sentence_t[0:149])
    train_s.append(s)

valid_s=[]
valid_t=[]
for sentence,sentence_t in zip(valid_sentences,valid_taggings):
    s = ""
    for word in sentence[0:149]:
        s = s +" "+ word
    valid_t.append(sentence_t[0:149])
    valid_s.append(s)


test_s=[]
test_t=[]
for sentence,sentence_t in zip(test_sentences,test_taggings):
    s = ""
    for word in sentence[0:149]:
        s = s +" "+ word
    test_t.append(sentence_t[0:149])
    test_s.append(s)



max_seq_len = max(max(map(len,map(methodcaller("split", " "), test_s))),max(map(len,map(methodcaller("split", " "), valid_s))),max(map(len,map(methodcaller("split", " "), train_s))))


def create_bert_embedding(dataset,sentences):
    print('Calculate BERT embedding for ' + dataset + "....")
    word_dict = bert.get_embd(sentences)
    #with open("persistent_files/bert_embedding_"+dataset+".txt", "wb") as fp:
    #    pickle.dump(word_dict, fp)
    return word_dict

word_embedding_t,word_embedding_v,word_embedding_test,tokens_test = [],[],[],[]

def embd_train():
    print("Embeddings train...")
    word_embedding_t,_=create_word(create_bert_embedding("training",train_s))
    return word_embedding_t


def embd_valid():
    print("Embeddings valid...")
    word_embedding_v,_=create_word(create_bert_embedding("valid",valid_s))
    return word_embedding_v


def embd_test():
    print("Embeddings test...")
    word_embedding_test,tokens_test=create_word(create_bert_embedding("test",test_s))
    return word_embedding_test,tokens_test


def avg(l1,l2):
    avg=[]
    for e1,e2 in zip(l1,l2):
        avg.append((e1+e2)/2)
    return avg

def sum_embd(l1,l2):
    avg=[]
    for e1,e2 in zip(l1,l2):
        avg.append(e1+e2)
    return avg

def max_embd(l1,l2):
    avg=[]
    for e1,e2 in zip(l1,l2):
        avg.append(max(e1,e2))
    return avg

def min_embd(l1,l2):
    avg=[]
    for e1,e2 in zip(l1,l2):
        avg.append(min(e1,e2))
    return avg

def fill(word_embedding,l,feature_len):
    temp = l-len(word_embedding)
    while temp>0:
        new = [0] * feature_len
        word_embedding.append(new)
        temp-=1
    return word_embedding

def create_word(bert_response):
    sentences_embd = []
    seq_len= len(bert_response[0][0])
    feature_len = len(bert_response[0][0][0])
    w,t = bert_response[0].tolist(),bert_response[1]
    for embd_sentence,token_sentence in zip(w,t):
        embd_sentence.pop(0)
        token_sentence.pop(0)
        del token_sentence[-1]
    for embd_sentence,token_sentence in zip(w,t):
        words_embedding = []
        for embd,token in itertools.zip_longest(embd_sentence,token_sentence,fillvalue="FILL"):
            if token.startswith("##",0,len(token)):
                words_embedding[-1]=avg(words_embedding[-1],embd)
            else:
                words_embedding.append(embd)
        words_embedding = fill(words_embedding,seq_len,feature_len)
        sentences_embd.append(words_embedding)
    return numpy.asarray(sentences_embd,dtype=numpy.float32),numpy.asarray(t)



if config["NEW_BERT_EMBD"]:
    print("Starting BERT server")
    start_bert.start_server(max_seq_len,config["PRETRAINED_MODEL"])
    word_embedding_t=embd_train()
    word_embedding_v=embd_valid()
    word_embedding_test,tokens_test=embd_test()
    start_bert.stop_server()
else:
    with open("persistent_files/bert_embedding_training.txt", "rb") as fp:
        word_embedding_t,_ = create_word(pickle.load(fp))
    with open("persistent_files/bert_embedding_valid.txt", "rb") as fp:
        word_embedding_v,_ = create_word(pickle.load(fp))
    with open("persistent_files/bert_embedding_test.txt", "rb") as fp:
        word_embedding_test,tokens_test =create_word(pickle.load(fp))


train_steps = (len(train_sentences) + config["BATCH_SIZE"] - 1) // config["BATCH_SIZE"]
valid_steps = (len(valid_sentences) + config["BATCH_SIZE"] - 1) // config["BATCH_SIZE"]


def batch_generator(sentences, taggings, steps,dataset,training=True):
    while True:
        for i in range(steps):
            #batch_sentences = sentences[config["BATCH_SIZE"] * i:min(config["BATCH_SIZE"] * (i + 1), len(sentences))]
            batch_taggings = taggings[config["BATCH_SIZE"] * i:min(config["BATCH_SIZE"] * (i + 1), len(taggings))]
            if dataset == "training":
                word_input = word_embedding_t[config["BATCH_SIZE"] * i:min(config["BATCH_SIZE"] * (i + 1), len(word_embedding_t))]
            if dataset== "valid":
                word_input= word_embedding_v[config["BATCH_SIZE"] * i:min(config["BATCH_SIZE"] * (i + 1), len(word_embedding_v))]
            if dataset=="test":
                word_input = word_embedding_test[config["BATCH_SIZE"] * i:min(config["BATCH_SIZE"] * (i + 1), len(word_embedding_test))]
            if not training:
                yield word_input, batch_taggings
                continue
            sentence_len = word_input.shape[1]
            batch_taggings_len = len(batch_taggings)
            for j in range(len(batch_taggings)):
                batch_taggings[j] = batch_taggings[j] + [0] * (sentence_len - len(batch_taggings[j]))
                batch_taggings[j] = [[tag] for tag in batch_taggings[j]]
            try:
                batch_taggings = numpy.asarray(batch_taggings,dtype=numpy.float32)
            except:
                for temp,sentence in enumerate(batch_taggings):
                    if len(sentence)!=150:
                        print(taggings)
                        print("temp:{0}".format(temp))
                        print("error:{0}".format(len(sentence)))
                        print("sentence len:{0}".format(sentence_len))
                        print("i: {0}".format(i))
                        print("len taggins: {0}".format(len(taggings)))
                        print("batch taggins: {0}".format(batch_taggings_len))
                        print("batch taggins j: {0}".format(batch_taggings_len_j))
                        print("dataset:{0}".format(dataset))
                        #print("sentence error len:{0}".format(len(batch_sentences[59])))
            batch_taggings = numpy.asarray(batch_taggings,dtype=numpy.float32)
            yield word_input, batch_taggings
        if not training:
            break


model = build_model(tag_num=len(TAGS),max_seq_len=max_seq_len)
model.summary(line_length=80)

if not(config["NEW_TRAINING"]):
    model.load_weights(config["BERT_MODEL"], by_name=True)
else:
    print('Fitting...')
    for lr in [1e-3,1e-4,1e-5]:
        model.fit_generator(
            generator=batch_generator(train_s, train_t, train_steps,"training"),
            steps_per_epoch=train_steps,
            epochs=config["EPOCHS"],
            validation_data=batch_generator(valid_s, valid_t, valid_steps,"valid"),
            validation_steps=valid_steps,
            callbacks=[
                keras.callbacks.EarlyStopping(monitor='val_acc', patience=5),
                ],
                verbose=True,
                )

        model.save_weights(config["BERT_MODEL"])


def get_tags(tags):
    filtered = []
    for i in range(len(tags)):
        if tags[i] == 0:
            filtered.append({
                'begin': i,
                'end': i,
                'type': 0,
            })
        if tags[i] % 2 == 1:
            filtered.append({
                'begin': i,
                'end': i,
                'type': i,
            })
        elif i > 0 and tags[i - 1] == tags[i] - 1:
            filtered[-1]['end'] += 1
    return filtered


def same_attr(tag_pred,tag_true):
    if tag_pred % 2 == 0:
        return tag_pred-1==tag_true
    else:
        return tag_pred+1==tag_true


print('Predicting...')

pr=True
if pr:
    eps = 1e-6
    total_pred, total_true, matched_num = 0, 0, 0.0
    total_fp,total_tp,total_fn,total_tn=0,0,0,0
    lines = []
    for inputs, batch_taggings in batch_generator(
            test_s,
            test_t,
            test_steps,
            "test",
            training=False):
        predict = model.predict_on_batch(inputs)
        predict = numpy.argmax(predict, axis=2).tolist()
        for i, pred in enumerate(predict):
            for tag in pred[:len(batch_taggings[i])]:
                for name,num in TAGS.items():
                    if num == tag:
                        t = name
                        lines.append("\t"+t+"\n")
            pred = pred[:len(batch_taggings[i])]
            true = batch_taggings[i]
            tn,tp,fn,fp=0,0,0,0
            for tag_pred,tag_true in zip(pred,true):
                if tag_pred!=tag_true and (not same_attr(tag_pred,tag_true)) and tag_pred!=0:
                    fp+=1
                if tag_pred!=0 and (tag_pred==tag_true or same_attr(tag_pred,tag_true)):
                    tp+=1
                if tag_true!=tag_pred and tag_pred==0:
                    fn+=1
                if tag_pred==tag_true and tag_pred==0:
                    tn+=1
            total_tp+=tp
            total_fp+=fp
            total_tn+=tn
            total_fn+=fn
            #print(str(tp)+" "+str(fp)+" "+str(tn)+" "+str(fn)+" ")
            #print("tp: "+str(total_tp)+"fp: "+str(total_fp)+"tn: "+str(total_tn)+"fn: "+str(total_fn))
            #print("\n")
            #matched_num += sum([1 for tag in pred if tag in true and tag["type"]!=0])
    precision = total_tp / (total_tp + total_fp+eps)
    recall = total_tp/ (total_tp + total_fn+eps)
    #recall = (matched_num + eps) / (total_true + eps)
    f1 = 2 * precision * recall / (precision + recall+eps)
    print('P: %.4f  R: %.4f  F: %.4f' % (precision, recall, f1))
    with open(config["DATA_ROOT"]+"/"+config["TEST_SET"],"r") as f:
        tokens = []
        for line in [l for l in f if not l.startswith("-DOCSTART-")]:
            words = utils.tokenizer(line)
            if words:
                word = words[0]
                tokens.append(word)
        f.close()
    with open("risultati_opentag/esperimento#"+str(config["ESPERIMENTO"])+".txt","w+") as f:
        for token,tag in zip(tokens,lines):
            f.write(token+tag)
        f.close()
        config["ESPERIMENTO"]+=1
        with open("config.json", "w") as c:
            json.dump(config,c)
