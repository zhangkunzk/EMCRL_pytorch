"""
@Author: zhkun
@Time:  21:39
@File: data_preprocessing
@Description: pre-processing operation for the dataset
@Something to attention
"""
import json
import os.path
import random

import pandas
import math
import nltk
import re
import gzip, pickle
import matplotlib.pyplot as plt
import csv
from tqdm import tqdm
import pandas as pd

import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.model.word_stats as nmw
from nlpaug.util import Action


offline = '/sentence_pair'
model_path = '/data/pretrained_models/aug_models'

wiki_line = '/SentEval/wiki'

split_deli = ';\t;'
aug_deli = ';\t--\t;'

three_labels = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
two_labels = {'entailment': 0, 'neutral': 1}
bi_labels = {'no': 0, 'yes': 1}


def load_data(path='train.csv'):
    label = []
    sentence1 = []
    sentence2 = []
    with open(path, 'r', newline='', encoding='utf-8') as f:
        for idx, line in tqdm(enumerate(f)):
            if idx == 0:
                continue
            cols = line.strip().split('\t')

            if cols[4] == '-':
                continue

            sentence1.append(cols[1])
            sentence2.append(cols[2])
            label.append(cols[4].lower())

    print(f'> processing all data from {path}, total count is {len(label)}')
    return [label, sentence1, sentence2]


def load_json_data(path='train.jsonl'):
    label = []
    sentence1 = []
    sentence2 = []
    with open(path, 'r', encoding='utf-8') as f:
        for idx, line in tqdm(enumerate(f)):
            example = json.loads(line)
            s1 = example['sentence1']
            s2 = example['sentence2']
            gold = example['gold_label']

            if gold not in three_labels.keys():
                continue

            sentence1.append(s1)
            sentence2.append(s2)
            label.append(gold.lower())

    print(f'> processing all data from {path}, total count is {len(label)}')
    return [label, sentence1, sentence2]


def load_text_data(path='train.txt'):
    label = []
    sentence1 = []
    sentence2 = []
    with open(path, 'r', newline='', encoding='utf-8') as f:
        for idx, line in tqdm(enumerate(f)):
            cols = line.strip().split('\t')

            gold = cols[0]
            if gold not in ['0', '1']:
                continue

            s1 = cols[1]
            s2 = cols[1]

            sentence1.append(s1)
            sentence2.append(s2)
            label.append(gold.lower())

    print(f'> processing all data from {path}, total count is {len(label)}')
    return [label, sentence1, sentence2]


def load_wiki_data(path):
    label = []
    sentence1 = []
    with open(path, 'r', newline='', encoding='utf-8') as f:
        for idx, line in tqdm(enumerate(f)):
            col = line.strip()
            sentence1.append(col)
            label.append('wiki')

    print(f'> processing all data from {path}, total count is {len(label)}')
    return [label, sentence1]


def load_dataset(data_name='sick'):
    if data_name == 'sick':
        train_path = os.path.join(offline, data_name, 'SICK_train.txt')
        dev_path = os.path.join(offline, data_name, 'SICK_trial.txt')
        test_path = os.path.join(offline, data_name, 'SICK_test_annotated.txt')
        load_func = load_data
    elif data_name == 'scitail':
        train_path = os.path.join(offline, data_name, 'scitail_1.0_train.txt')
        dev_path = os.path.join(offline, data_name, 'scitail_1.0_dev.txt')
        test_path = os.path.join(offline, data_name, 'scitail_1.0_test.txt')
        load_func = load_json_data
    elif data_name == 'quora':
        train_path = os.path.join(offline, data_name, 'train.tsv')
        dev_path = os.path.join(offline, data_name, 'dev.tsv')
        test_path = os.path.join(offline, data_name, 'test.tsv')
        load_func = load_json_data
    elif data_name == 'wiki':
        data_path = os.path.join(wiki_line, 'wiki1m_for_simcse.txt')
        load_func = load_wiki_data
    else:
        raise ValueError ('Wrong parameters, please try again')

    if data_name != 'wiki':
        train_data = load_func(train_path)
        dev_data = load_func(dev_path)
        test_data = load_func(test_path)
        return train_data, test_data, dev_data
    else:
        train_data = load_func(data_path)
        return train_data


def simple_clean_sentence(sent):
    punctuation = ['.', ',', '!', '/', ':', ';',
                   '+', '-', '*', '?', '~', '|',
                   '[', ']', '{', '}', '(', ')',
                   '_', '=', '%', '&', '$', '#',
                   '"', '`', '^']

    sent = sent.replace('\n', ' ').replace('\\n', ' ').replace('\\', ' ')
    for p in punctuation:
        sent = sent.replace(p, ' ' + p + ' ')
    # sent = re.sub(r'\d+\.?\d*', ' numbernumbernumbernumbernumber ', sent)
    return sent.lower()


def extract_text_and_labels(d):
    """
    :param d:
    :return:
    """
    ret = {"label": [], "sentence1": [], "sentence2": []}
    for idx in range(len(d[0])):
        if type(d[2][idx]) is float and math.isnan(d[2][idx]):
            continue
        cleaned1 = simple_clean_sentence(d[1][idx]).lower()
        cleaned2 = simple_clean_sentence(d[2][idx]).lower()
        ret['sentence2'].append(cleaned2)
        ret["label"].append(d[0][idx])
        ret['sentence1'].append(cleaned1)

    return ret


def sentence_augment(sentence, number=5, max_len=100):
    #
    #
    aug_methods = [
        nac.RandomCharAug(action='substitute'),
        naw.WordEmbsAug(model_type='word2vec',
                        model_path=os.path.join(model_path, 'GoogleNews-vectors-negative300.bin'), action='insert'),
        naw.WordEmbsAug(model_type='word2vec',
                        model_path=os.path.join(model_path, 'GoogleNews-vectors-negative300.bin'), action='substitute'),
        naw.WordEmbsAug(model_type='fasttext',
                        model_path=os.path.join(model_path, 'wiki-news-300d-1M.vec'), action='insert'),
        naw.WordEmbsAug(model_type='fasttext',
                        model_path=os.path.join(model_path, 'wiki-news-300d-1M.vec'), action='substitute'),
        naw.TfIdfAug(model_path=model_path, action='insert'),
        naw.TfIdfAug(model_path=model_path, action='substitute'),
        naw.SynonymAug(aug_src='wordnet'),
        naw.SynonymAug(aug_src='ppdb',
                       model_path=os.path.join(model_path, 'ppdb-2.0-l-all')),
        naw.RandomWordAug(action="swap"),
        naw.RandomWordAug(),
        naw.RandomWordAug(action='crop'),
        naw.SplitAug()
    ]

    results = []
    for idx in range(number):
        method_id = random.randint(0, len(aug_methods))
        aug_sen = aug_methods[method_id].augment(sentence)
        aug_sen = nltk.word_tokenize(aug_sen)
        if len(aug_sen) > max_len:
            aug_sen = ' '.join(aug_sen[:max_len])
        else:
            aug_sen = ' '.join(aug_sen)
        results.append(aug_sen)
    results = aug_deli.join(results)

    results = aug_deli.join(results)

    return results


def generated_augments(d, number=10, max_len=100):
    ret = {"label": [], "sentence1": [], "sentence2": [], "sentence1_aug": [], "sentence2_aug": []}

    for idx in range(len(d['sentence1'])):
        s1_aug = sentence_augment(d['sentence1'][idx], number=number, max_len=max_len)
        ret['sentence1'].append(d['sentence1'][idx])
        ret['sentence1_aug'].append(s1_aug)
        ret['label'].append(d['label'][idx])
        if 'sentence2' in d.keys():
            s2_aug = sentence_augment(d['sentence2'][idx], number=number, max_len=max_len)
            ret['sentence2'].append(d['sentence2'][idx])
            ret['sentence2_aug'].append(s2_aug)

    for item in ret.keys():
        if len(ret[item]) == 0:
            ret.pop(item)

    return ret


def word_tokenize(d, max_len=100):
    for idx in range(len(d['sentence1'])):
        d['sentence1'][idx] = nltk.word_tokenize(d['sentence1'][idx])
        d['sentence2'][idx] = nltk.word_tokenize(d['sentence2'][idx])

    ret = {"sentence1": [], "sentence2": [], "label": []}
    for idx in range(len(d["sentence1"])):
        if len(d["sentence1"][idx]) <= max_len and len(d["sentence2"][idx]) <= max_len:
            ret["sentence1"].append(' '.join(d["sentence1"][idx]))
            ret["sentence2"].append(' '.join(d["sentence2"][idx]))
            ret["label"].append(d["label"][idx])
        elif len(d["sentence1"][idx]) > max_len:
            ret["sentence1"].append(' '.join(d["sentence1"][idx][:max_len]))
            ret["sentence2"].append(' '.join(d["sentence2"][idx]))
            ret["label"].append(d["label"][idx])
        else:
            ret["sentence1"].append(' '.join(d["sentence1"][idx]))
            ret["sentence2"].append(' '.join(d["sentence2"][idx][:max_len]))
            ret["label"].append(d["label"][idx])
    return ret


def create_tfidf_file(d):
    sentences = []
    if 'sentence1' in d.keys():
        sentences += [item.split(' ') for item in d['sentence1']]
    elif 'sentence2' in d.keys():
        sentences += [item.split(' ') for item in d['sentence2']]
    else:
        raise ValueError('not necessary keys, please check again')

    tfidf_model = nmw.TfIdf()
    tfidf_model.train(sentences)
    tfidf_model.save(model_path=model_path)


def save_dataset(d, path="yahoo_answers_test.pkl.gz"):
    f = gzip.open(path, "wb")
    pickle.dump(d, f)
    f.close()


def save_csv(d, path="train_clean2.csv"):
    keys = list(d.keys())
    # keys = keys[:2]
    try:
        with open(path, 'w', encoding='utf8') as f:
            f.write(split_deli.join(keys)+'\n')
            # writer = csv.DictWriter(f, fieldnames=keys)
            # writer.writeheader()
            ##
            for idx in range(len(d[keys[0]])):
                if len(keys) == 3:
                    row = split_deli.join([d[keys[0]][idx], d[keys[1]][idx], d[keys[2]][idx]])
                elif len(keys) == 2:
                    row = split_deli.join([d[keys[0]][idx], d[keys[1]][idx]])
                elif len(keys) == 5:
                    row = split_deli.join([d[keys[0]][idx], d[keys[1]][idx], d[keys[2]][idx], d[keys[3]][idx], d[keys[4]][idx]])
                else:
                    raise ValueError('wrong keys number, please check again')
                # row = {keys[0]: d[keys[0]][idx], keys[1]: d[keys[1]][idx]}
                f.write(row+'\n')
    except IOError:
        print("I/O error")


def stat_word(d):
    stat = {}

    for s in d["text"]:
        for w in s:
            if w in stat.keys():
                stat[w] += 1
            else:
                stat[w] = 1
    stat = sorted(stat.items(), key=lambda d: d[1], reverse=True)

    return stat


def stat_sentence_length(d):
    stat = {}

    for s in d["text"]:
        l = len(s)
        if l in stat.keys():
            stat[l] += 1
        else:
            stat[l] = 1
    stat = sorted(stat.items(), key=lambda d: d[0], reverse=True)

    return stat


def stat_label(d):
    stat = {}

    for l in d["label"]:
        if l in stat.keys():
            stat[l] += 1
        else:
            stat[l] = 1

    return stat


def generate_clean_csv(data_name='scitail', max_len=100):
    save_name = ['train_cleaned.csv', 'test_cleaned.csv', 'dev_cleaned.csv']

    contents = load_dataset(data_name)

    for idx, data in enumerate(contents):
        extracted = extract_text_and_labels(data)
        d_tr = word_tokenize(extracted, max_len)
        save_csv(d_tr, path=os.path.join(offline, data_name, data_name+'_'+save_name[idx]))
        print(f'> pre-processing finishend: save file {data_name}-{save_name[idx]}')


def generate_aug_csv(data_name='scitail', number=10, max_len=100):
    save_name = ['train_aug.csv', 'test_aug.csv', 'dev_aug.csv']

    contents = load_dataset(data_name)

    if 'tfidfaug_w2idf.txt' not in os.listdir(os.path.join(offline, data_name)):
        flag = True
    else:
        flag = False

    for idx, data in enumerate(contents):
        extracted = extract_text_and_labels(data)
        d_tr = word_tokenize(extracted, max_len)
        if flag:
            create_tfidf_file(d_tr)
            flag = False
        d_aug = generated_augments(d_tr, number=number, max_len=max_len)
        save_csv(d_aug, path=os.path.join(offline, data_name, data_name+'_'+save_name[idx]))
        print(f'> pre-processing finishend: save file {data_name}-{save_name[idx]}')


def generate_stat_word(tr=None, te=None, d=None,
                       train_path="yahoo_answers_train.pkl.gz",
                       test_path="yahoo_answers_test.pkl.gz",
                       dict_path="yahoo_answers_dict.pkl.gz"):
    if (tr is None or te is None) and d is None:
        d_te = load_dataset(path="yahoo_answers_test.pkl.gz")
        print("Test set loaded!")
        d_tr = load_dataset(path="yahoo_answers_train.pkl.gz")
        print("Train set loaded!")

    if d is None:
        d = {"text": [], "label": [], "lookup_table": []}
        d["lookup_table"] = d_tr["lookup_table"]
        d["text"] = d_tr["text"] + d_te["text"]
        d["label"] = d_tr["label"] + d_te["label"]

    s_word = stat_word(d)
    f = open("word_stat.txt", "w", encoding="UTF-8")
    for inst in s_word:
        f.write(str(inst[1]) + "\t" + inst[0] + "\n")
    f.close()

    f = gzip.open(dict_path, "wb")
    pickle.dump(s_word, f)
    f.close()

    return s_word, d


def generate_stat_sentence_length(tr=None, te=None, d=None,
                                  train_path="yahoo_answers_train.pkl.gz",
                                  test_path="yahoo_answers_test.pkl.gz"):
    if (tr is None or te is None) and d is None:
        d_te = load_dataset(path="yahoo_answers_test.pkl.gz")
        print("Test set loaded!")
        d_tr = load_dataset(path="yahoo_answers_train.pkl.gz")
        print("Train set loaded!")

    if d is None:
        d = {"text": [], "label": [], "lookup_table": []}
        d["lookup_table"] = d_tr["lookup_table"]
        d["text"] = d_tr["text"] + d_te["text"]
        d["label"] = d_tr["label"] + d_te["label"]

    s_senlen = stat_sentence_length(d)

    count = [i[1] for i in s_senlen]
    length = [i[0] for i in s_senlen]
    plt.plot(length, count, 'ro')
    plt.savefig("len_distribution.png")
    plt.show()

    return s_senlen, d


def generate_stat_label(tr=None, te=None, d=None,
                        train_path="yahoo_answers_train.pkl.gz",
                        test_path="yahoo_answers_test.pkl.gz"):
    if (tr is None or te is None) and d is None:
        d_te = load_dataset(path="yahoo_answers_test.pkl.gz")
        print("Test set loaded!")
        d_tr = load_dataset(path="yahoo_answers_train.pkl.gz")
        print("Train set loaded!")

    if d is None:
        d = {"text": [], "label": [], "lookup_table": []}
        d["lookup_table"] = d_tr["lookup_table"]
        d["text"] = d_tr["text"] + d_te["text"]
        d["label"] = d_tr["label"] + d_te["label"]

    s_label = stat_label(d)
    s_label = s_label.items()

    count = [i[1] for i in s_label]
    length = [i[0] for i in s_label]
    plt.plot(length, count, 'ro')
    plt.savefig("label_distribution.png")
    plt.show()

    return s_label, d


if __name__ == "__main__":

    data_name = 'wiki'
    max_length = 100

    contents = load_dataset(data_name)
    dataset = contents[1]
    extracted = extract_text_and_labels(dataset)
    d_tr = word_tokenize(extracted, max_length)

    # create_tfidf_file(d_tr)
    d_aug = generated_augments(d_tr, number=5, max_len=max_length)
    save_csv(d_aug, path=os.path.join(offline, data_name, data_name + '_simple_test_agu.csv'))
    # print(f'> pre-processing finishend: save file {data_name}-{save_name[idx]}')
    generate_aug_csv(data_name, number=5, max_len=max_length)
