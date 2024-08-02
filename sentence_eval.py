"""
@Author: zhkun
@Time:  21:39
@File: sentval
@Description: SentEval toolkit for the trained model.
@Something to attention
"""
import sys
import io, os
import numpy as np
# import logging
import argparse
from prettytable import PrettyTable
import torch
import transformers
from transformers import AutoModel, AutoTokenizer
from torch import nn
from model import EMCL
from my_parser import test_parser, parser
from utils import write_file
# Set up logger
# logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

# Set PATHs
PATH_TO_SENTEVAL = 'D:/SentEval'
PATH_TO_DATA = 'D:/SentEval'

if not os.path.exists(PATH_TO_SENTEVAL):
    PATH_TO_SENTEVAL = '/data/SentEval'
    PATH_TO_DATA = '/data/SentEval'

# Import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval


def load_model(model, path, device):
    ckpt_path = path
    print('Load checkpoint', ckpt_path)
    checkpoint = torch.load(ckpt_path, map_location=device)
    try:
        model.load_state_dict(checkpoint['state_dict'])
        # checkpoint = {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}
        # model.load_state_dict(checkpoint)
    except:
        # if saving a paralleled model but loading an unparalleled model
        model = nn.DataParallel(model)
        model.load_state_dict(checkpoint['state_dict'])

    print(f'> model at {ckpt_path} is loaded!')


def print_table(task_names, scores, file_name=None):
    tb = PrettyTable()
    tb.field_names = task_names
    tb.add_row(scores)
    if file_name is not None:
        write_file(file_name, tb)
    print(tb)


class current_pase(object):
    def __init__(self):
        self.model_name_or_path = 'bert-base-uncased'
        # {cls, cls_before_pooler, avg, avg_topn,avg_first_last, first_last, last_topn}
        self.pooler = 'avg_first_last'
        # ['dev', 'test', 'fasttest']
        self.mode = 'dev'
        # ['sts', 'transfer', 'full', 'na']
        self.task_set = 'sts'
        self.tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC', 'MRPC',
                      'SICKRelatedness', 'STSBenchmark']


def single_eval(args, my_pas, model, tokenizer, device, save_file=None):
    # Set up the tasks
    if args.task_set == 'sts':
        args.tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']
    elif args.task_set == 'transfer':
        args.tasks = ['MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC', 'MRPC']
    elif args.task_set == 'full':
        args.tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']
        args.tasks += ['MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC', 'MRPC']

    # Set params for SentEval
    if args.mode == 'dev' or args.mode == 'fasttest':
        # Fast mode
        params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
        params['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                'tenacity': 3, 'epoch_size': 2}
    elif args.mode == 'test':
        # Full mode
        params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10}
        params['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64,
                                'tenacity': 5, 'epoch_size': 4}
    else:
        raise NotImplementedError

    # SentEval prepare and batcher
    def prepare(params, samples):
        return

    def batcher(params, batch, max_length=None):
        # Handle rare token encoding issues in the dataset
        if len(batch) >= 1 and len(batch[0]) >= 1 and isinstance(batch[0][0], bytes):
            batch = [[word.decode('utf-8') for word in s] for s in batch]

        sentences = [' '.join(s) for s in batch]

        # Tokenization
        if max_length is not None:
            batch = tokenizer.batch_encode_plus(
                sentences,
                return_tensors='pt',
                padding=True,
                max_length=max_length,
                truncation=True
            )
        else:
            batch = tokenizer.batch_encode_plus(
                sentences,
                return_tensors='pt',
                padding=True,
            )

        # Move to the correct device
        for k in batch:
            batch[k] = batch[k].to(device)

        if my_pas.use_origin:
            # Get raw embeddings
            with torch.no_grad():
                outputs = model(**batch, output_hidden_states=True, return_dict=True)
                last_hidden = outputs.last_hidden_state
                pooler_output = outputs.pooler_output
                hidden_states = outputs.hidden_states
        else:
            with torch.no_grad():
                outputs = model(batch)
                # bert_output2相关的内容都没用
                if my_pas.use_predict:
                    cls_emb1, z1, bert_outputs1, prediction = outputs
                else:
                    cls_emb1, z1, bert_outputs1 = outputs

                last_hidden = bert_outputs1.last_hidden_state
                pooler_output = bert_outputs1.pooler_output
                hidden_states = bert_outputs1.hidden_states

        # Apply different poolers
        if args.pooler == 'cls_before_pooler':
            # There is a linear+activation layer after CLS representation
            return last_hidden[:, 0].cpu()
        # elif args.pooler == 'cls_before_pooler':
        #     return last_hidden[:, 0].cpu()
        elif args.pooler == "avg":
            return ((last_hidden * batch['attention_mask'].unsqueeze(-1)).sum(1) / batch['attention_mask'].sum(
                -1).unsqueeze(-1)).cpu()
        elif args.pooler == "avg_first_last":
            first_hidden = hidden_states[0]
            last_hidden = hidden_states[-1]
            pooled_result = ((first_hidden + last_hidden) / 2.0 * batch['attention_mask'].unsqueeze(-1)).sum(
                1) / batch['attention_mask'].sum(-1).unsqueeze(-1)
            return pooled_result.cpu()
        # elif args.pooler == "avg_top2":
        #     second_last_hidden = hidden_states[-2]
        #     last_hidden = hidden_states[-1]
        #     pooled_result = ((last_hidden + second_last_hidden) / 2.0 * batch['attention_mask'].unsqueeze(
        #         -1)).sum(1) / batch['attention_mask'].sum(-1).unsqueeze(-1)
        #     return pooled_result.cpu()
        elif args.pooler in ['cls', 'first_last', 'last_topn']:
            if my_pas.use_word_attention:
                return cls_emb1.cpu()
            else:
                if args.pooler == 'cls':
                    return last_hidden[:, 0].cpu()
                elif args.pooler == 'first_last':
                    results = ((hidden_states[0]+hidden_states[-1])/2.0 * batch['attention_mask'].unsqueeze(-1)).sum(1) / \
                              batch['attention_mask'].sum(-1).unsqueeze(-1)
                    return results.cpu()
                else:
                    results = torch.stack(hidden_states[-my_pas.num_hidden_layers:], dim=2)
                    results = torch.mean(results, dim=2)
                    results = (results * batch['attention_mask'].unsqueeze(-1)).sum(1) / batch['attention_mask'].sum(
                        -1).unsqueeze(-1)
                    return results.cpu()
        else:
            raise NotImplementedError

    results = {}

    for task in args.tasks:
        se = senteval.engine.SE(params, batcher, prepare)
        result = se.eval(task)
        results[task] = result

    # Print evaluation results
    if args.mode == 'dev':
        print("------ %s ------" % (args.mode))

        task_names = []
        scores = []
        for task in ['STSBenchmark', 'SICKRelatedness']:
            task_names.append(task)
            if task in results:
                scores.append("%.2f" % (results[task]['dev']['spearman'][0] * 100))
            else:
                scores.append("0.00")
        print_table(task_names, scores, save_file)

        task_names = []
        scores = []
        for task in ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']:
            task_names.append(task)
            if task in results:
                scores.append("%.2f" % (results[task]['devacc']))
            else:
                scores.append("0.00")
        task_names.append("Avg.")
        scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
        print_table(task_names, scores, save_file)

    elif args.mode == 'test' or args.mode == 'fasttest':
        print("------ %s ------" % (args.mode))

        task_names = []
        scores = []
        for task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']:
            task_names.append(task)
            if task in results:
                if task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16']:
                    scores.append("%.2f" % (results[task]['all']['spearman']['all'] * 100))
                else:
                    scores.append("%.2f" % (results[task]['test']['spearman'].correlation * 100))
            else:
                scores.append("0.00")
        task_names.append("Avg.")
        scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
        print_table(task_names, scores, save_file)

        task_names = []
        scores = []
        for task in ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']:
            task_names.append(task)
            if task in results:
                scores.append("%.2f" % (results[task]['acc']))
            else:
                scores.append("0.00")
        task_names.append("Avg.")
        scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
        print_table(task_names, scores, save_file)


def evaluation(save_model_type, my_pas=None):
    if my_pas is None:
        my_pas = parser()
    args = current_pase()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    # my_pas.name = my_pas.name+save_model_type

    if my_pas.use_origin:
        # Load transformers' model checkpoint
        model = AutoModel.from_pretrained(args.model_name_or_path, cache_dir=my_pas.cache_dir)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, cache_dir=my_pas.cache_dir)
        model = model.to(device)
    else:
        # 改成自己的模型
        model = EMCL(my_pas)
        model.to(device)
        # model.eval()
        tokenizer = AutoTokenizer.from_pretrained(my_pas.pre_trained_model, cache_dir=my_pas.cache_dir)
        ckpt_path = os.path.join('checkpoint', '{}.pth'.format(my_pas.name+save_model_type))
        load_model(model, ckpt_path, device)

    result_file = os.path.join(my_pas.log_path, f'evaluation_{my_pas.name}{save_model_type}.txt')
    # pooler_type = ['cls', 'cls_before_pooler', 'avg', 'avg_first_last', 'first_last', 'last_topn']
    pooler_type = ['cls', 'cls_before_pooler', 'first_last']
    # modes = ['dev', 'test', 'fasttest']
    # task_set = ['sts', 'transfer', 'full']
    task = 'full'
    mode = 'test'
    # for task in task_set:
    #     for mode in modes:
    for pooler in pooler_type:
        args.task_set = task
        args.mode = mode
        args.pooler_type = pooler
        seperate_line = '\n' + '=='*50 + '\n' + ' '*20+task+'-'*5+mode+'--'*10+pooler + '\n'+'=='*50 + '\n'
        write_file(result_file, seperate_line)

        single_eval(
            args=args,
            my_pas=my_pas,
            model=model,
            tokenizer=tokenizer,
            device=device,
            save_file=result_file
        )


def main():
    args = current_pase()
    my_pas = test_parser()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if my_pas.use_origin:
        # Load transformers' model checkpoint
        model = AutoModel.from_pretrained(args.model_name_or_path, cache_dir=my_pas.cache_dir)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, cache_dir=my_pas.cache_dir)
        model = model.to(device)
    else:
        # 改成自己的模型
        model = EMCL(my_pas)
        model.to(device)
        # model.eval()
        tokenizer = AutoTokenizer.from_pretrained(my_pas.pre_trained_model, cache_dir=my_pas.cache_dir)
        ckpt_path = os.path.join('checkpoint', '{}.pth'.format(my_pas.name))
        load_model(model, ckpt_path, device)

    # Set up the tasks
    if args.task_set == 'sts':
        args.tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']
    elif args.task_set == 'transfer':
        args.tasks = ['MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC', 'MRPC']
    elif args.task_set == 'full':
        args.tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']
        args.tasks += ['MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC', 'MRPC']

    # Set params for SentEval
    if args.mode == 'dev' or args.mode == 'fasttest':
        # Fast mode
        params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
        params['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                'tenacity': 3, 'epoch_size': 2}
    elif args.mode == 'test':
        # Full mode
        params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10}
        params['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64,
                                'tenacity': 5, 'epoch_size': 4}
    else:
        raise NotImplementedError

    # SentEval prepare and batcher
    def prepare(params, samples):
        return

    def batcher(params, batch, max_length=None):
        # Handle rare token encoding issues in the dataset
        if len(batch) >= 1 and len(batch[0]) >= 1 and isinstance(batch[0][0], bytes):
            batch = [[word.decode('utf-8') for word in s] for s in batch]

        sentences = [' '.join(s) for s in batch]

        # Tokenization
        if max_length is not None:
            batch = tokenizer.batch_encode_plus(
                sentences,
                return_tensors='pt',
                padding=True,
                max_length=max_length,
                truncation=True
            )
        else:
            batch = tokenizer.batch_encode_plus(
                sentences,
                return_tensors='pt',
                padding=True,
            )

        # Move to the correct device
        for k in batch:
            batch[k] = batch[k].to(device)

        if my_pas.use_origin:
            # Get raw embeddings
            with torch.no_grad():
                outputs = model(input_ids=batch['input_ids'],
                                attention_mask=batch['attention_mask'],
                                token_type_ids=batch['token_type_ids'],
                                output_hidden_states=True,
                                return_dict=True)
                last_hidden = outputs.last_hidden_state
                pooler_output = outputs.pooler_output
                hidden_states = outputs.hidden_states
        else:
            with torch.no_grad():
                outputs = model(batch)
                # outputs = model(
                #     input_ids=batch['input_ids'],
                #     attention_mask=batch['attention_mask'],
                #     token_type_ids=batch['token_type_ids']
                # )
                # bert_output2相关的内容都没用
                if my_pas.use_predict:
                    cls_emb1, z1, bert_outputs1, prediction = outputs
                else:
                    cls_emb1, z1, bert_outputs1 = outputs

                last_hidden = bert_outputs1.last_hidden_state
                pooler_output = bert_outputs1.pooler_output
                hidden_states = bert_outputs1.hidden_states

        # Apply different poolers
        if args.pooler == 'cls_before_pooler':
            # There is a linear+activation layer after CLS representation
            return last_hidden[:, 0].cpu()
        # elif args.pooler == 'cls_before_pooler':
        #     return last_hidden[:, 0].cpu()
        elif args.pooler == "avg":
            return ((last_hidden * batch['attention_mask'].unsqueeze(-1)).sum(1) / batch['attention_mask'].sum(
                -1).unsqueeze(-1)).cpu()
        elif args.pooler == "avg_first_last":
            first_hidden = hidden_states[0]
            last_hidden = hidden_states[-1]
            pooled_result = ((first_hidden + last_hidden) / 2.0 * batch['attention_mask'].unsqueeze(-1)).sum(
                1) / batch['attention_mask'].sum(-1).unsqueeze(-1)
            return pooled_result.cpu()
        elif args.pooler == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = ((last_hidden + second_last_hidden) / 2.0 * batch['attention_mask'].unsqueeze(
                -1)).sum(1) / batch['attention_mask'].sum(-1).unsqueeze(-1)
            return pooled_result.cpu()
        elif args.pooler in ['cls', 'first_last', 'last_topn', 'avg_topn']:
            if my_pas.use_word_attention:
                return cls_emb1.cpu()
            else:
                results = torch.stack(hidden_states[-my_pas.num_hidden_layers:], dim=2)
                results = torch.mean(results, dim=2)
                results = (results * batch['attention_mask'].unsqueeze(-1)).sum(1) / batch['attention_mask'].sum(-1).unsqueeze(-1)
                return results.cpu()
        else:
            raise NotImplementedError

    results = {}

    for task in args.tasks:
        se = senteval.engine.SE(params, batcher, prepare)
        result = se.eval(task)
        results[task] = result

    # Print evaluation results
    if args.mode == 'dev':
        print("------ %s ------" % (args.mode))

        task_names = []
        scores = []
        for task in ['STSBenchmark', 'SICKRelatedness']:
            task_names.append(task)
            if task in results:
                scores.append("%.2f" % (results[task]['dev']['spearman'][0] * 100))
            else:
                scores.append("0.00")
        print_table(task_names, scores)

        task_names = []
        scores = []
        for task in ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']:
            task_names.append(task)
            if task in results:
                scores.append("%.2f" % (results[task]['devacc']))
            else:
                scores.append("0.00")
        task_names.append("Avg.")
        scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
        print_table(task_names, scores)

    elif args.mode == 'test' or args.mode == 'fasttest':
        print("------ %s ------" % (args.mode))

        task_names = []
        scores = []
        for task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']:
            task_names.append(task)
            if task in results:
                if task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16']:
                    scores.append("%.2f" % (results[task]['all']['spearman']['all'] * 100))
                else:
                    scores.append("%.2f" % (results[task]['test']['spearman'].correlation * 100))
            else:
                scores.append("0.00")
        task_names.append("Avg.")
        scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
        print_table(task_names, scores)

        task_names = []
        scores = []
        for task in ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']:
            task_names.append(task)
            if task in results:
                scores.append("%.2f" % (results[task]['acc']))
            else:
                scores.append("0.00")
        task_names.append("Avg.")
        scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
        print_table(task_names, scores)


if __name__ == "__main__":
    # main()
    # model_type = ['last', 'test']
    # for item in model_type:
    #     evaluation(item)
    evaluation_type = ['test', 'test_loss', 'last']
    for single_type in evaluation_type:
        # print(PATH_TO_SENTEVAL)
        evaluation(single_type)
        try:
            evaluation(single_type)
        except FileNotFoundError:
            continue
