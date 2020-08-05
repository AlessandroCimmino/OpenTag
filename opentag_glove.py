import os
import codecs
import numpy
import keras
from keras_wc_embd import get_dicts_generator, get_batch_input
from models.model import build_model
import re
import pre_tagging as p
import bert
import json
import utils

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


root = config["DATA_ROOT"]
DATA_TRAIN_PATH = os.path.join(root, 'train_set.txt')
DATA_VALID_PATH = os.path.join(root, 'valid_set.txt')
DATA_TEST_PATH = os.path.join(root, config["TEST_SET"])


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


dicts_generator = get_dicts_generator(
    word_min_freq=2,
    char_min_freq=1e100,
    word_ignore_case=True,
    char_ignore_case=False
)
for sentence in train_sentences:
    dicts_generator(sentence)
word_dict, _, _ = dicts_generator(return_dict=True)


if os.path.exists(config["GLOVE_WORD_EMBD"]):
    word_dict = {
        '': 0,
        'UNK': 1,
    }
    word_embd_weights = [
        [0.0] * 100,
        numpy.random.random((100,)).tolist(),
    ]
    with codecs.open(config["GLOVE_WORD_EMBD"], 'r', 'utf8') as reader:
        for line_num, line in enumerate(reader):
            if (line_num + 1) % 1000 == 0:
                print('Load embedding... %d' % (line_num + 1), end='\r', flush=True)
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            word = parts[0].lower()
            if word not in word_dict:
                word_dict[word] = len(word_dict)
                word_embd_weights.append(parts[1:])
    word_embd_weights = numpy.asarray(word_embd_weights)
    print('Dict size: %d  Shape of weights: %s' % (len(word_dict), str(word_embd_weights.shape)))
else:
    word_embd_weights = None
    print('Dict size: %d' % len(word_dict))


train_steps = (len(train_sentences) + config["BATCH_SIZE"] - 1) // config["BATCH_SIZE"]
valid_steps = (len(valid_sentences) + config["BATCH_SIZE"] - 1) // config["BATCH_SIZE"]

def batch_generator(sentences, taggings, steps, training=True):
    global word_dict
    while True:
        for i in range(steps):
            batch_sentences = sentences[config["BATCH_SIZE"] * i:min(config["BATCH_SIZE"] * (i + 1), len(sentences))]
            batch_taggings = taggings[config["BATCH_SIZE"] * i:min(config["BATCH_SIZE"] * (i + 1), len(taggings))]
            word_input, _ = get_batch_input(
                batch_sentences,
                1,
                word_dict,
                {},
                word_ignore_case=True,
                char_ignore_case=False
            )
            if not training:
                yield word_input, batch_taggings
                continue
            sentence_len = word_input.shape[1]
            for j in range(len(batch_taggings)):
                batch_taggings[j] = batch_taggings[j] + [0] * (sentence_len - len(batch_taggings[j]))
                batch_taggings[j] = [[tag] for tag in batch_taggings[j]]
            batch_taggings = numpy.asarray(batch_taggings)
            yield word_input, batch_taggings
        if not training:
            break


model = build_model(token_num=len(word_dict),
                    tag_num=len(TAGS))
model.summary(line_length=80)

if not(config["NEW_TRAINING"]):
    model.load_weights(config["GLOVE_MODEL"], by_name=True)
else:
    print('Fitting...')
    for lr in [1e-3,1e-4,1e-5]:
        model.fit_generator(
            generator=batch_generator(train_sentences, train_taggings, train_steps),
            steps_per_epoch=train_steps,
            epochs=config["EPOCHS"],
            validation_data=batch_generator(valid_sentences, valid_taggings, valid_steps),
            validation_steps=valid_steps,
            callbacks=[
                keras.callbacks.EarlyStopping(monitor='val_acc', patience=5),
                ],
                verbose=True,
                )

        model.save_weights(config["GLOVE_MODEL"])

test_sentences, test_taggings = load_data(DATA_TEST_PATH)
test_steps = (len(test_sentences) + config["BATCH_SIZE"] - 1) // config["BATCH_SIZE"]

print('Predicting...')


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


pr=True
if pr:
    eps = 1e-6
    total_pred, total_true, matched_num = 0, 0, 0.0
    total_fp,total_tp,total_fn,total_tn=0,0,0,0
    lines = []
    for inputs, batch_taggings in batch_generator(
            test_sentences,
            test_taggings,
            test_steps,
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
    with open(root+"/"+config["TEST_SET"],"r") as f:
        tokens = []
        for line in [l for l in f if not l.startswith("-DOCSTART-")]:
            words = utils.tokenizer(line)
            if words:
                word = words[0]
                tokens.append(word)
        f.close()
    with open("risultati_opentag/esperimento#"+str(config["ESPERIMENTO"])+".txt","w+") as f:
        f.truncate(0)
        for token,tag in zip(tokens,lines):
            f.write(token+tag)
        f.close()
        config["ESPERIMENTO"]+=1
        with open("config.json", "w") as c:
            json.dump(config,c)
