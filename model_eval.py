"""
@Author: zhkun
@Time:  15:22
@File: evaluate_case
@Description: 直接输出得到结果的热力图和相似度值
@Something to attention
"""
import os.path

import senteval
import torch
import numpy as np

from transformers import AutoTokenizer

from model import EMCL
from torch import nn
from utils import write_file
# from prettytable import PrettyTable
from my_parser import parser

# Set PATHs
PATH_TO_SENTEVAL = 'D:/data/SentEval'
PATH_TO_DATA = 'D:/data/SentEval'

if not os.path.exists(PATH_TO_SENTEVAL):
    PATH_TO_SENTEVAL = '/data/SentEval'
    PATH_TO_DATA = '/data/SentEval'


def load_model(model, path, device):
    ckpt_path = path
    print('Load checkpoint', ckpt_path)
    checkpoint = torch.load(ckpt_path, map_location=device)
    try:
        model.load_state_dict(checkpoint['state_dict'])
    except:
        # if saving a paralleled model but loading an unparalleled model
        model = nn.DataParallel(model)
        model.load_state_dict(checkpoint['state_dict'])

    print(f'> model at {ckpt_path} is loaded!')


# def print_table(task_names, scores, file_name=None):
#     tb = PrettyTable()
#     tb.field_names = task_names
#     tb.add_row(scores)
#     if file_name is not None:
#         write_file(file_name, tb)
#     print(tb)


def single_eval(args, my_pas, model, tokenizer, device, save_file=None):
    # Set up the tasks

    # if args.task_set == 'sts':
    #     args.tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']
    # elif args.task_set == 'transfer':
    #     args.tasks = ['MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC', 'MRPC']
    # elif args.task_set == 'full':
    #     args.tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']
    #     args.tasks += ['MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC', 'MRPC']

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

        #
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

    # return {'devacc': devacc, 'acc': testacc,
    #          'ndev': len(task_embed['dev']['X']),
    #          'ntest': len(task_embed['test']['X'])}

    # evaluate_set = args.tasks
    evaluate_set = args.current_task

    for task in evaluate_set:
        se = senteval.engine.SE(params, batcher, prepare)
        result = se.eval(task)
        results[task] = result

    # Print evaluation results
    if args.mode == 'dev':
        print("------ %s ------" % (args.mode))

        task_names = []
        scores = []
        for task in ['STSBenchmark', 'SICKRelatedness']:
            if task in results:
                task_names.append(task)
                scores.append("%.2f" % (results[task]['dev']['spearman'][0] * 100))
        # print_table(task_names, scores, save_file)

        task_names = []
        scores = []
        for task in ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']:
            if task in results:
                task_names.append(task)
                scores.append("%.2f" % (results[task]['devacc']))
        if len(scores) > 0:
            task_names.append("Avg.")
            scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
        # print_table(task_names, scores, save_file)

    elif args.mode == 'test' or args.mode == 'fasttest':
        print("------ %s ------" % (args.mode))

        task_names = []
        scores = []
        for task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']:

            if task in results:
                task_names.append(task)
                if task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16']:
                    scores.append("%.2f" % (results[task]['all']['spearman']['all'] * 100))
                else:
                    scores.append("%.2f" % (results[task]['test']['spearman'].correlation * 100))
        if len(scores) > 0:
            task_names.append("Avg.")
            scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
        # print_table(task_names, scores, save_file)

        task_names = []
        scores = []
        for task in ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']:
            if task in results:
                task_names.append(task)
                scores.append("%.2f" % (results[task]['acc']))
        if len(scores) > 0:
            task_names.append("Avg.")
            scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
        # print_table(task_names, scores, save_file)

    return results


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
        self.current_task = ['STSBenchmark', 'SICKRelatedness']
        # evaluation_metric: 'spearman', 'pearson'
        self.evaluate_metric = 'spearman'


def evaluation(current_model=None, evaluate_type=None, tokenizer=None, device=None, my_pas=None):
    if my_pas is None:
        my_pas = parser()

    current_parameters = current_pase()

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(current_parameters.model_name_or_path, cache_dir=my_pas.cache_dir)

    if current_model is None:
        current_model = EMCL(my_pas)
        current_model.to(device)
        ckpt_path = os.path.join('checkpoint', '{}.pth'.format(my_pas.name+evaluate_type))
        load_model(current_model, ckpt_path, device)

    result_file = os.path.join(my_pas.log_path, f'evaluation_{my_pas.name}{evaluate_type}.txt')
    pooler_type = ['cls', 'first_last']
    task = current_parameters.task_set
    mode = current_parameters.mode

    best_stsb_dresults = {}
    best_stsb_tresults = {}

    for pooler in pooler_type:
        current_parameters.pooler_type = pooler
        seperate_line = '\n' + '==' * 50 + '\n' + ' ' * 20 + task + '-' * 5 + mode + '--' * 10 + pooler + '\n' + '==' * 50 + '\n'
        write_file(result_file, seperate_line)

        results = single_eval(
                args=current_parameters,
                my_pas=my_pas,
                model=current_model,
                tokenizer=tokenizer,
                device=device,
                save_file=result_file
        )

        for dt in current_parameters.current_task:
            dev_value = results[dt]['dev'][current_parameters.evaluate_metric].correlation * 100
            test_value = results[dt]['test'][current_parameters.evaluate_metric].correlation * 100
            if dt not in best_stsb_dresults.keys():
                best_stsb_dresults[dt] = [dev_value]
            else:
                best_stsb_dresults[dt].append(dev_value)

            if dt not in best_stsb_tresults.keys():
                best_stsb_tresults[dt] = [test_value]
            else:
                best_stsb_tresults[dt].append(test_value)

    best_dev_acc = {}
    best_test_acc = {}

    for dt in current_parameters.current_task:
        best_dev_acc[dt] = np.max(np.asarray(best_stsb_dresults[dt]))
        best_test_acc[dt] = np.max(np.asarray(best_stsb_tresults[dt]))

    return best_dev_acc, best_test_acc


def test(my_pas=None):
    # args = current_pase()
    if my_pas is None:
        my_pas = test_parser()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    evaluate_type = 'test_loss'

    model = EMCL(my_pas)
    model.to(device)
    # model.eval()
    tokenizer = AutoTokenizer.from_pretrained(my_pas.pre_trained_model, cache_dir=my_pas.cache_dir)
    ckpt_path = os.path.join('checkpoint', '{}.pth'.format(my_pas.name+evaluate_type))
    load_model(model, ckpt_path, device)

    results = evaluation(
        current_model=model,
        evaluate_type=evaluate_type,
        tokenizer=tokenizer,
        device=device,
        my_pas=my_pas
    )

    print(results)


if __name__ == '__main__':
    print(f'> Start to test the model')
    my_pas = parser()

    test(my_pas)











