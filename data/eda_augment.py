"""
@Author: zhkun
@Time:  17:47
@File: eda_augment
@Description: 使用EDA的方法来生成增强的数据
@Something to attention
"""

from augment import gen_eda
from eda import eda
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



offline = '/sentence_pair'
model_path = '/aug_models'

wiki_line = '/SentEval/wiki'

split_deli = ';\t;'
aug_deli = ';\t--\t;'

three_labels = {'entailment': 0, 'neutral': 1, 'contradiction': 2}
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
    elif data_name == 'snli':
        train_path = os.path.join(offline, data_name, 'snli_1.0_train.jsonl')
        dev_path = os.path.join(offline, data_name, 'snli_1.0_dev.jsonl')
        test_path = os.path.join(offline, data_name, 'snli_1.0_test.jsonl')
        hard_path = os.path.join(offline, data_name, 'snli_1.0_test_hard.jsonl')
        load_func = load_json_data
    elif data_name == 'wiki':
        data_path = os.path.join(wiki_line, 'wiki1m_for_simcse.txt')
        load_func = load_wiki_data
    else:
        raise ValueError('Wrong parameters, please try again')

    if data_name == 'snli':
        train_data = load_func(train_path)
        dev_data = load_func(dev_path)
        test_data = load_func(test_path)
        hard_data = load_func(hard_path)
        return [train_data, test_data, dev_data, hard_data]
    elif data_name in ['scitail', 'sick', 'quora']:
        train_data = load_func(train_path)
        dev_data = load_func(dev_path)
        test_data = load_func(test_path)
        return [train_data, test_data, dev_data]
    else:
        wi_data = load_func(data_path)
        length = len(wi_data[0])
        split_point = int(length * 0.9)
        train_data = [item[:split_point] for item in wi_data]
        test_data = [item[split_point:] for item in wi_data]
        return [train_data, test_data]


def simple_clean_sentence(line):
    # punctuation = ['.', ',', '!', '/', ':', ';',
    #                '+', '-', '*', '?', '~', '|',
    #                '[', ']', '{', '}', '(', ')',
    #                '_', '=', '%', '&', '$', '#',
    #                '"', '`', '^']
    #
    # line = line.replace('\n', ' ').replace('\\n', ' ').replace('\\', ' ')
    # for p in punctuation:
    #     line = line.replace(p, ' ' + p + ' ')
    # # # sent = re.sub(r'\d+\.?\d*', ' numbernumbernumbernumbernumber ', sent)

    clean_line = ""
    line = line.replace("’", "")
    line = line.replace("'", "")
    line = line.replace("-", " ")  # replace hyphens with spaces
    line = line.replace("\t", " ")
    line = line.replace("\n", " ")
    line = line.lower()

    # 去除所有的非英文字符和中文字符
    pattern = u'[^0-9a-zA-Z\u4e00-\u9fa5.，,。？“”]+'
    line = re.sub(pattern, ' ', line)
    if len(line) <= 0:
        print(line)

    for char in line:
        if char in 'qwertyuiopasdfghjklzxcvbnm ,.?':
            clean_line += char
        else:
            clean_line += ' '

    clean_line = re.sub(' +', ' ', clean_line)  # delete extra spaces
    if len(clean_line) == 1:
        clean_line = ''
    elif clean_line[0] == ' ':
        clean_line = clean_line[1:]

    # pattern = r"(?u)\b\w\w+\b"
    # token_pattern = re.compile(pattern)
    # result = ' '.join(token_pattern.findall(clean_line))
    return clean_line.lower()


# def extract_text_and_labels(d):
#     """
#     :param d:
#     :return:
#     """
#     ret = {"label": [], "sentence1": [], "sentence2": []}
#     for idx in range(len(d[0])):
#         if type(d[2][idx]) is float and math.isnan(d[2][idx]):
#             continue
#         cleaned1 = simple_clean_sentence(d[1][idx]).lower()
#         cleaned2 = simple_clean_sentence(d[2][idx]).lower()
#         if len(cleaned1) == 0 or len(cleaned2) == 0:
#             continue
#         if type(cleaned1) is float and math.isnan(cleaned1):
#             continue
#         if type(cleaned2) is float and math.isnan(cleaned2):
#             continue
#         ret['sentence2'].append(cleaned2)
#         ret["label"].append(d[0][idx])
#         ret['sentence1'].append(cleaned1)
#
#     return ret


def extract_text_and_labels(d):
    """
    :param d:
    :return:
    """
    if len(d) == 2:
        ret = {"label": [], "sentence1": []}
        for idx in range(len(d[0])):
            if type(d[1][idx]) is float and math.isnan(d[1][idx]):
                continue
            cleaned1 = simple_clean_sentence(d[1][idx]).lower()
            if len(cleaned1) <= 1 or len(cleaned1.split(' ')) < 5:
                continue
            if type(cleaned1) is float and math.isnan(cleaned1):
                continue

            ret["label"].append(d[0][idx])
            ret['sentence1'].append(cleaned1)
    else:
        ret = {"label": [], "sentence1": [], "sentence2": []}
        for idx in range(len(d[0])):
            if type(d[2][idx]) is float and math.isnan(d[2][idx]):
                continue
            cleaned1 = simple_clean_sentence(d[1][idx]).lower()
            cleaned2 = simple_clean_sentence(d[2][idx]).lower()
            if len(cleaned1) <= 1 or len(cleaned1.split(' ')) < 5 or len(cleaned2) <= 1 or len(cleaned2.split(' ')) < 5:
                continue
            if type(cleaned1) is float and math.isnan(cleaned1):
                continue
            if type(cleaned2) is float and math.isnan(cleaned2):
                continue
            ret['sentence2'].append(cleaned2)
            ret["label"].append(d[0][idx])
            ret['sentence1'].append(cleaned1)

    return ret


# def create_single_files(d, data_name, type='train'):
#     """
#     生成符合要求的单个文件，用于支持EDA方法
#     :param d:
#     :return:
#     """
#     for item in ['sentence1', 'sentence2']:
#         content = []
#         if item not in d.keys():
#             continue
#         for idx in range(len(d['label'])):
#             line = '\t'.join([d['label'][idx], d[item][idx]])
#             content.append(line)
#
#         if data_name == 'wiki':
#             path = os.path.join(wiki_line, item+'_'+type+'_aug.temp')
#         else:
#             path = os.path.join(offline, data_name, item + '_' + type + '_aug.temp')
#         with open(path, 'w', encoding='utf8') as f:
#             for idx in range(len(content)):
#                 if idx == len(content) - 1:
#                     f.write(content[idx])
#                 else:
#                     f.write(content[idx]+'\n')


def generated_augment_file(args, d, data_name, sentence_type=None, type='train'):
    alpha_sr = args['alpha_sr'] # number of augmented sentences per original sentence
    alpha_ri = args['alpha_ri'] # percent of words in each sentence to be replaced by synonyms
    alpha_rs = args['alpha_rs'] # percent of words in each sentence to be inserted
    alpha_rd = args['alpha_rd'] # percent of words in each sentence to be swapped
    num_aug = args['num_aug'] # percent of words in each sentence to be deleted

    if data_name == 'wiki':
        base_path = wiki_line
    else:
        base_path = os.path.join(offline, data_name)

    if sentence_type is None:
        sentence_type = ['sentence1']
    for item in sentence_type:
        print(f'> processing sentence type {item} in {data_name}')
        output_file = open(os.path.join(base_path, item+'_'+type+'_aug.txt'), 'w', encoding='utf-8')

        for idx in tqdm(range(len(d['label']))):
            label = d['label'][idx]
            origin_sentence = d[item][idx]
            try:
                # print(f'> {idx}--{origin_sentence}')
                aug_sentences = eda(origin_sentence,
                                    alpha_sr=alpha_sr,
                                    alpha_ri=alpha_ri,
                                    alpha_rs=alpha_rs,
                                    p_rd=alpha_rd,
                                    num_aug=num_aug
                                    )
            except IndexError:
                print(origin_sentence)
                continue

            output_file.write(label + "\t" + aug_sentences + '\n')

        output_file.close()


def generate_aug_data(d, data_name, type='train'):
    ret = {"label": [], "sentence1": [], "sentence2": [], "sentence1_aug": [], "sentence2_aug": []}

    if data_name == 'wiki':
        base_path = wiki_line
    else:
        base_path = os.path.join(offline, data_name)
    f1 = open(os.path.join(base_path, 'sentence1_'+type+'_aug.txt'), 'r', encoding='utf8')
    f1_lines = f1.readlines()

    assert len(f1_lines) == len(d['label'])

    if 'sentence2' in d.keys():
        f2 = open(os.path.join(base_path, 'sentence2_' + type + '_aug.txt'), 'r', encoding='utf8')
        f2_lines = f2.readlines()
        assert len(f1_lines) == len(d['label'])

    for idx in range(len(d['label'])):
        s1_aug = f1_lines[idx].strip().split('\t')[1:]
        s1_aug = aug_deli.join(s1_aug)
        ret['sentence1'].append(d['sentence1'][idx])
        ret['sentence1_aug'].append(s1_aug)
        ret['label'].append(d['label'][idx])
        if 'sentence2' in d.keys():
            s2_aug = f2_lines[idx].strip().split('\t')[1:]
            s2_aug = aug_deli.join(s2_aug)
            ret['sentence2'].append(d['sentence2'][idx])
            ret['sentence2_aug'].append(s2_aug)

    f1.close()
    os.remove(os.path.join(base_path, 'sentence1_'+type+'_aug.txt'))
    if 'sentence2' in d.keys():
        f2.close()
        os.remove(os.path.join(base_path, 'sentence2_' + type + '_aug.txt'))

    key_list = list(ret.keys())
    for item in key_list:
        if len(ret[item]) == 0:
            ret.pop(item)

    return ret


def word_tokenize(d, max_len=100):
    if 'sentence2' not in d.keys():
        for idx in range(len(d['sentence1'])):
            d['sentence1'][idx] = nltk.word_tokenize(d['sentence1'][idx])

        ret = {"sentence1": [], "label": []}
        for idx in range(len(d["sentence1"])):
            if len(d["sentence1"][idx]) <= max_len:
                ret["sentence1"].append(' '.join(d["sentence1"][idx]))
                ret["label"].append(d["label"][idx])
            elif len(d["sentence1"][idx]) > max_len:
                ret["sentence1"].append(' '.join(d["sentence1"][idx][:max_len]))
                ret["label"].append(d["label"][idx])
            else:
                raise ValueError('wrong operation, please try again')
    else:
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


def generate_aug_csv(data_name='scitail', number=10, max_len=100):
    data_type = ['train', 'test', 'dev', 'hard']
    save_name = ['train_aug.csv', 'test_aug.csv', 'dev_aug.csv', 'test_hard_aug.csv']

    args = {
        'alpha_sr': 0.3,  # how much to replace each word by synonyms
        'alpha_ri': 0.1, # how much to insert new words that are synonyms
        'alpha_rs': 0.1, # how much to swap words
        'alpha_rd': 0.05, # how much to delete words
        'num_aug': number # number of augmented sentences per original sentence
    }

    contents = load_dataset(data_name)

    for idx, data in enumerate(contents):
        print(f'processing file {data_type[idx]}')
        extracted = extract_text_and_labels(data)
        d_tr = word_tokenize(extracted, max_len)
        # create_single_files(d_tr, data_name, type=data_type[idx])
        if 'sentence2' in d_tr.keys():
            generated_augment_file(args, d_tr, data_name, sentence_type=['sentence1', 'sentence2'], type=data_type[idx])
        else:
            generated_augment_file(args, d_tr, data_name, sentence_type=['sentence1'], type=data_type[idx])

        d_aug = generate_aug_data(d_tr, data_name, type=data_type[idx])
        if data_name == 'wiki':
            save_csv(d_aug, path=os.path.join(wiki_line, data_name+'_'+save_name[idx]))
        else:
            save_csv(d_aug, path=os.path.join(offline, data_name, data_name + '_' + save_name[idx]))
        print(f'> pre-processing finishend: save file {data_name}-{save_name[idx]}')


if __name__ == "__main__":

    generate_aug_csv(data_name='scitail', number=10, max_len=100)
    exit()

    data_type = ['train', 'test', 'dev', 'hard']
    save_name = ['train_aug.csv', 'test_aug.csv', 'dev_aug.csv', 'test_hard_aug.csv']

    data_name = 'scitail'
    max_length = 100

    args = {
        'alpha_sr': 0.3,  # how much to replace each word by synonyms
        'alpha_ri': 0.1,  # how much to insert new words that are synonyms
        'alpha_rs': 0.1,  # how much to swap words
        'alpha_rd': 0.,  # how much to delete words
        'num_aug': 5  # number of augmented sentences per original sentence
    }

    test_idx = 1
    contents = load_dataset(data_name)
    dataset = contents[test_idx]
    extracted = extract_text_and_labels(dataset)
    d_tr = word_tokenize(extracted, max_length)

    if 'sentence2' in d_tr.keys():
        generated_augment_file(args, d_tr, data_name, sentence_type=['sentence1', 'sentence2'], type=data_type[test_idx])
    else:
        generated_augment_file(args, d_tr, data_name, sentence_type=['sentence1'], type=data_type[test_idx])

    d_aug = generate_aug_data(d_tr, data_name, type=data_type[test_idx])
    if data_name == 'wiki':
        save_csv(d_aug, path=os.path.join(wiki_line, data_name + '_' + save_name[test_idx]))
    else:
        save_csv(d_aug, path=os.path.join(offline, data_name, data_name + '_' + save_name[test_idx]))
    print(f'> pre-processing finishend: save file {data_name}-{save_name[test_idx]}')

