"""
@Author: zhkun
@Time:  21:33
@File: solver
@Description: 模型的主要接口
@Something to attention
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import numpy as np
import os
import pickle as pkl

from transformers import AdamW
from transformers import AutoTokenizer
import time
import random
import datetime

import photinia as ph
from utils import get_current_time, calc_eplased_time_since
from model import EMCL
from data.dataset import WikiData, SentimentData
from utils import eval_map_mrr, BiClassCalculator, write_file, CosineWarmUpDecay
from utils import nt_xent_loss, nt_em_loss, nt_cl_loss, uniform_loss, align_loss, align_uniform_loss
from evaluate_case import write_csv
from model_eval import evaluation


class Solver:
    def __init__(self, args):
        # how to use GPUs
        # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        num_workers = max([4 * torch.cuda.device_count(), 4])

        train_loader = None
        dev_loader = None
        test_loader = None
        test_hard_loader = None

        torch.manual_seed(args.seed)
        if args.data_name.lower() in ['snli']:
            dataset = SentimentData(args)
        elif args.data_name.lower() == 'wiki':
            dataset = WikiData(args)
        else:
            raise ValueError(f'wrong dataset name {args.data_name.lower()}, please try again')

        if args.data_name.lower() == 'snli':
            train_loader, test_loader, dev_loader, test_hard_loader = dataset.get_dataloaders(
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=device == 'cuda'
            )
            print('#examples:',
                  '\n#train', len(train_loader.dataset),
                  '\n#dev', len(dev_loader.dataset),
                  '\n#test', len(test_loader.dataset),
                  '\n#hard', len(test_hard_loader.dataset)
                  )
        else:
            train_loader, test_loader = dataset.get_dataloaders(
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=device == 'cuda'
            )
            print('#examples:',
                  '\n#train', len(train_loader.dataset),
                  '\n#test', len(test_loader.dataset))

        self.tokenizer = AutoTokenizer.from_pretrained(
            args.pre_trained_model,
            do_lower_case=True,
            cache_dir=args.cache_dir
        )

        if args.net == 'emcl':
            model = EMCL(args)
        else:
            raise ValueError('wrong net name, please try again')

        device_count = 0
        if device == 'cuda':
            device_count = torch.cuda.device_count()
            if device_count > 1:
                model = nn.DataParallel(model)
            torch.backends.cudnn.benchmark = True
            print("Let's use {} GPUs!".format(device_count))
        model.to(device)
        self.device = device
        self.vocab_size = model.module.bert.config.vocab_size if device_count > 1 else model.bert.config.vocab_size

        # Other optimizer
        params = model.module.req_grad_params if device_count > 1 else model.req_grad_params
        optimizer = optim.Adam(params, lr=args.lr, betas=(0.9, 0.999), amsgrad=True, weight_decay=args.weight_decay)

        # Bert optimizer
        param_optimizer = list(model.module.bert.named_parameters() if device_count > 1 else model.bert.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]
        if args.do_mlm:
            param_optimizer1 = list(
                model.module.lm_head.named_parameters() if device_count > 1 else model.lm_head.named_parameters())
            optimizer_grouped_parameters1 = [
                {'params': [p for n, p in param_optimizer1 if not any(nd in n for nd in no_decay)],
                 'weight_decay_rate': 0.01},
                {'params': [p for n, p in param_optimizer1 if any(nd in n for nd in no_decay)],
                 'weight_decay_rate': 0.0}
            ]
            optimizer_grouped_parameters += optimizer_grouped_parameters1

        optimizer_bert = AdamW(optimizer_grouped_parameters, lr=3e-5)

        if args.cl_loss == 'default':
            cl_loss = nt_cl_loss
        elif args.cl_loss == 'only_true':
            cl_loss = nt_xent_loss
        elif args.cl_loss == 'em':
            cl_loss = nt_em_loss
        else:
            raise ValueError('wrong parameters, please check again')

        if args.align_uniform:
            align_loss = align_uniform_loss
            self.align_loss = align_loss

        classify_loss = nn.CrossEntropyLoss()

        # args.name += '_bert' if args.train_bert else ''
        ckpt_path = os.path.join('checkpoint', '{}'.format(args.name))
        if not os.path.exists(ckpt_path+'.pth'):
            print('Not found ckpt', ckpt_path)

        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.optimizer_bert = optimizer_bert
        self.cl_loss = cl_loss
        self.classify_loss = classify_loss
        self.device = device
        self.dataset = dataset
        self.ckpt_path = ckpt_path
        self.train_loader = train_loader

        self.dev_loader = dev_loader
        self.test_loader = test_loader
        self.test_hard_loader = test_hard_loader

        self.batch_idx = 0
        self.training_log = []
        self.testing_log = []

    def train(self):
        print('Starting Traing....')
        best_loss = float('inf')
        best_test_loss = float('inf')
        best_test_meric = 0.
        best_meric = 0.

        self.big_epochs = len(self.train_loader.dataset) // (self.args.batch_size * self.args.accumulate_step)

        self.learning_scheduler = ph.optim.lr_scheduler.CosineWarmUpAnnealingLR(
            optimizer=self.optimizer,
            num_loops=self.args.epochs * self.big_epochs,
            min_factor=1e-8,
        )

        for epoch in range(1, self.args.epochs + 1):
            epoch_start_time = time.time()
            print('-' * 20 + 'Epoch: {}, {}'.format(epoch, get_current_time()) + '-' * 20)
            if epoch != 1:
                write_file(os.path.join(self.args.log_path, f'testing_log_{self.args.name}'), self.testing_log)
                self.testing_log.clear()

            train_loss = self.train_epoch(epoch)
            if self.args.data_name == 'wiki':
                dev_meric, test_meric = self.wiki_evaluate('dev')
                if dev_meric > best_meric:
                    best_meric = dev_meric
                    self.save_model('dev')
            else:
                dev_meric = self.evaluate_epoch('Dev')
                test_meric = self.evaluate_epoch('Test')

                if isinstance(dev_meric, list):
                    if dev_meric[1] > best_meric:
                        best_meric = dev_meric[1]
                        self.save_model('dev')
                    if dev_meric[0] < best_loss:
                        best_loss = dev_meric[0]
                        self.save_model('dev_loss')
                else:
                    if dev_meric < best_loss:
                        best_loss = dev_meric
                        self.save_model('dev_loss')

            if self.dev_loader is None:
                self.dev_loader = self.test_loader

            if self.args.use_predict:
                test_log = f'------------------{datetime.datetime.now()}----------------------------\t' \
                           f'Epoch:{epoch}\t' \
                           f'{self.args.name} \t' \
                           f'Train loss:{train_loss[0]:.5f}\t' \
                           f'Train acc:{train_loss[1]:.5f}\t' \
                           f'Dev Loss:{dev_meric[0]:.5f}' \
                           f'Test Loss:{test_meric[0]:.5f}, \t ' \
                           f'Dev acc:{dev_meric[1]:.5f}, \t ' \
                           f'Test acc:{test_meric[1]:.5f}, \t ' \
                           f'Best Dev Loss:{best_loss:.5f}, \t' \
                           f'Best Test Loss:{best_test_loss:.5f}, \t' \
                           f'Best Test acc:{best_meric:.5f}, \t' \
                           f'Best Test acc:{best_test_meric:.5f}, \t'
            elif self.args.data_name == 'wiki':
                test_log = f'------------------{datetime.datetime.now()}----------------------------\t' \
                           f'Epoch:{epoch}\t' \
                           f'{self.args.name} \t' \
                           f'Train loss:{train_loss:.5f}\t' \
                           f'Best Dev Metric:{best_meric:.5f}, \t' \
                           f'Best Test Metric:{best_test_meric:.5f}, \t'
            else:
                test_log = f'------------------{datetime.datetime.now()}----------------------------\t' \
                           f'Epoch:{epoch}\t' \
                           f'{self.args.name} \t' \
                           f'Train loss:{train_loss:.5f}\t' \
                           f'Dev Loss:{dev_meric:.5f}' \
                           f'Test Loss:{test_meric:.5f}, \t ' \
                           f'Best Dev Loss:{best_loss:.5f}, \t' \
                           f'Best Test Loss:{best_test_loss:.5f}, \t'

            self.testing_log.append(test_log)

            print(test_log.replace('\t', '\n'))

            write_file(os.path.join(self.args.log_path, f'training_log_{self.args.name}.log'), self.training_log)
            self.training_log.clear()

        print('Training Finished!')

        self.save_model(name='last')

        self.test()

    def test(self):
        # Load the best checkpoint
        test_logs = []
        for name in ['dev', 'dev_loss', 'last']:
            try:
                self.load_model(name)
            except FileNotFoundError:
                print(f'> model {name} is not found')
                continue

            if self.args.data_name == 'snli':
                print('*' * 25 + f'Final dev result at {name}' + '*' * 25)
                # print(f'Final dev result at {name}..............')
                test_loss = self.evaluate_epoch('Dev', name)
                if isinstance(test_loss, list):
                    log = f'Dev Loss: {test_loss[0]:.4f}, Dev acc: {test_loss[1]:.4f}'
                else:
                    log = 'Dev Loss: {:.3f}'.format(test_loss)
                print(log.replace('\t', '\n'))
                test_logs.append(log)

                # Test
                print('*' * 25 + f'Final test result at {name}' + '*' * 25)
                test_loss = self.evaluate_epoch('Test', name)
                # print('Test Loss: {:.3f}, Test Acc: {:.3f}'.format(test_loss, test_acc))
                if isinstance(test_loss, list):
                    log = f'Test Loss: {test_loss[0]:.4f}, Test acc: {test_loss[1]:.4f}'
                else:
                    log = 'Test Loss: {:.3f}'.format(test_loss)
                print(log.replace('\t', '\n'))
                test_logs.append(log)

                print('*' * 25 + f'Final test hard result at {name}' + '*' * 25)
                test_loss = self.evaluate_epoch('hard', name)
                # print('Test hard Loss: {:.3f}, Test Acc: {:.3f}'.format(test_loss, test_acc))
                if isinstance(test_loss, list):
                    log = f'Hard Loss: {test_loss[0]:.4f}, Hard acc: {test_loss[1]:.4f}'
                else:
                    log = 'Hard Loss: {:.3f}'.format(test_loss)
                print(log.replace('\t', '\n'))
                test_logs.append(log)
            else:
                # Test
                print('*' * 25 + f'Final test result at {name}' + '*' * 25)
                test_loss = self.evaluate_epoch('Test', name)
                # print('Test Loss: {:.3f}, Test Acc: {:.3f}'.format(test_loss, test_acc))
                if isinstance(test_loss, list):
                    log = f'Test Loss: {test_loss[0]:.4f}, Test acc: {test_loss[1]:.4f}'
                else:
                    log = 'Test Loss: {:.3f}'.format(test_loss)
                print(log.replace('\t', '\n'))
                test_logs.append(log)

        write_file(os.path.join(self.args.log_path, f'testing_log_{self.args.name}.log'), test_logs)

    def train_epoch(self, epoch_idx):
        self.model.train()
        train_loss = 0.
        correct = 0
        example_count = 0

        loader_iter = iter(self.train_loader)
        for batch_idx in range(self.big_epochs):
            step_loss = 0
            step_class_loss = 0
            step_mask_loss = 0
            step_correct = 0
            step_example_count = 0
            all_z1 = []
            all_z2 = []

            # earse all the gradient for the current loop
            self.optimizer.zero_grad()
            self.optimizer_bert.zero_grad()

            for step in range(self.args.accumulate_step):
                try:
                    content = next(loader_iter)
                except StopIteration:
                    loader_iter = iter(self.train_loader)
                    content = next(loader_iter)
                if self.args.do_mlm:
                    if len(content) == 7:
                        pairs_info = {'input_ids': content[0], 'attention_mask': content[1]}
                        aug_info = {'input_ids': content[2], 'attention_mask': content[3]}
                    elif len(content) == 9:
                        pairs_info = {'input_ids': content[0], 'token_type_ids': content[1], 'attention_mask': content[2]}
                        aug_info = {'input_ids': content[3], 'token_type_ids': content[4], 'attention_mask': content[5]}
                    else:
                        raise Exception

                    y = content[-3]
                    pair_mlm = content[-2]
                    mlm_labels = content[-1]
                else:
                    if len(content) == 5:
                        pairs_info = {'input_ids': content[0], 'attention_mask': content[1]}
                        aug_info = {'input_ids': content[2], 'attention_mask': content[3]}
                        y = content[-1]
                    elif len(content) == 7:
                        pairs_info = {'input_ids': content[0], 'token_type_ids': content[1], 'attention_mask': content[2]}
                        aug_info = {'input_ids': content[3], 'token_type_ids': content[4], 'attention_mask': content[5]}
                        y = content[-1]
                    else:
                        raise Exception

                pairs_info = {pair_key: pairs_info[pair_key].to(self.device) for pair_key in pairs_info.keys()}
                aug_info = {aug_key: aug_info[aug_key].to(self.device) for aug_key in aug_info}
                target = y.to(self.device)
                if self.args.do_mlm:
                    pair_mlm = pair_mlm.to(self.device)
                    mlm_labels = mlm_labels.to(self.device)
                else:
                    pair_mlm = None
                    mlm_labels = None

                if self.args.aug_type == 'default':
                    results_info = self.model(
                        pair_info=pairs_info,
                        aug_pair=pairs_info,
                        pair_mlm=pair_mlm
                    )
                else:
                    results_info = self.model(
                        pair_info=pairs_info,
                        aug_pair=aug_info,
                        pair_mlm=pair_mlm
                    )

                if self.args.use_predict:
                    if self.args.do_mlm:
                        cls_emb1, cls_emb2, z1, z2, bert_outputs1, bert_outputs2, prediction, mlm_prediction = results_info
                    else:
                        cls_emb1, cls_emb2, z1, z2, bert_outputs1, bert_outputs2, prediction = results_info
                else:
                    if self.args.do_mlm:
                        cls_emb1, cls_emb2, z1, z2, bert_outputs1, bert_outputs2, mlm_prediction = results_info
                    else:
                        cls_emb1, cls_emb2, z1, z2, bert_outputs1, bert_outputs2 = results_info

                all_z1.append(z1)
                all_z2.append(z2)

                if self.args.use_predict:
                    # all_prediction.append(prediction)
                    loss_class = self.args.classify_weight * self.classify_loss(prediction, target) / self.args.accumulate_step
                    # loss_class = self.args.classify_weight * self.classify_loss(prediction, target)
                    step_class_loss += loss_class.item()
                    # 这部分的模型和原来的模型使用的是一张图（那如果mlm用的不是一张图，训练还有意义么？）
                    loss_class.backward(retain_graph=True)

                    pred_c = torch.max(prediction, 1)[1]
                    correct_pred = pred_c.eq(target.view_as(pred_c)).sum().item()
                    step_correct += correct_pred
                    step_example_count += len(prediction)

                if self.args.do_mlm:
                    # masked_lm_loss = self.classify_loss(mlm_prediction.view(-1, self.vocab_size), mlm_labels.view(-1))
                    masked_lm_loss = self.args.mlm_weight * masked_lm_loss / self.args.accumulate_step
                    masked_lm_loss = self.args.mlm_weight * masked_lm_loss
                    step_mask_loss += masked_lm_loss.item()
                    masked_lm_loss.backward()

            all_z1 = torch.cat(all_z1, dim=0)
            all_z2 = torch.cat(all_z2, dim=0)

            if self.args.cl_loss == 'default':
                loss_cl = self.cl_loss(all_z1, all_z2, self.classify_loss, device=self.device, t1=self.args.cl_tempure)
            elif self.args.cl_loss == 'sem':
                # this loss function should provide supervised target matrix
                loss_cl = self.cl_loss(all_z1, all_z2, supervised_mat=None, t1=self.args.cl_tempure)
            elif self.args.cl_loss == 'em':
                loss_cl = self.cl_loss(all_z1, all_z2, t1=self.args.cl_tempure, noise_prob=self.args.noise_prob)
            else:
                loss_cl = self.cl_loss(all_z1, all_z2, t1=self.args.cl_tempure)

            loss_cl.backward()

            step_loss = loss_cl.item()

            if self.args.use_predict:
                step_loss += step_class_loss

            if self.args.do_mlm:
                # all_mlm_label = torch.cat(all_mlm_label, dim=0)
                # all_mlm_prediction = torch.cat(all_mlm_prediction, dim=0)
                # masked_lm_loss = self.args.mlm_weight * self.classify_loss(all_mlm_prediction, all_mlm_label)
                step_loss += step_mask_loss

            if self.args.grad_max_norm > 0.:
                torch.nn.utils.clip_grad_norm_(self.model.req_grad_params, self.args.grad_max_norm)
            self.optimizer.step()
            self.optimizer_bert.step()
            self.learning_scheduler.step()

            current_learning_rate = self.learning_scheduler.get_last_lr()[0]
            # for param_group in self.optimizer.param_groups:
            #     param_group['lr'] = self.learning_value.get_value(epoch_idx * self.big_epochs + batch_idx + 1)

            current_time = str(datetime.datetime.now()).split('.')[0]
            loss_screen = f'current_loss: {step_loss:.4f}\tcl_Loss:{loss_cl:.4f}\t'
            if self.args.do_mlm:
                loss_screen += f'mlm_loss:{step_mask_loss:.4f}\t'
            if self.args.use_predict:
                loss_screen += f'classify_loss:{step_class_loss:.4f}\tclassify_acc:{step_correct / step_example_count * 1.0:.4f}\t'
                correct += step_correct
                example_count += step_example_count

            screen_log = f'{current_time}\tBatch:{epoch_idx}--{batch_idx + 1}({self.args.batch_size * self.args.accumulate_step})\t' \
                         f'{self.args.name}\t'
            screen_log += loss_screen
            screen_log += f'lr:{current_learning_rate:.6f}'

            self.training_log.append(screen_log.replace('\t', ', '))

            if batch_idx == 0 or (batch_idx + 1) % self.args.display_step == 0:
                print(screen_log.replace('\t', ', '))
            # print(screen_log.replace('\t', ', '))

            train_loss += step_loss
        if self.args.use_predict:
            acc = correct / (example_count * 1.0)
            return [train_loss, acc]
        else:
            return train_loss

    def wiki_evaluate(self, mode, model_name='training'):
        print(f'Evaluating {mode} on STS-B....')
        self.model.eval()

        results = evaluation(
            current_model=self.model,
            evaluate_type=mode,
            tokenizer=self.tokenizer,
            device=self.device,
            my_pas=self.args
        )

        if self.args.eval_data == 'stsb':
            best_dev = results[0]['STSBenchmark']
            best_acc = results[1]['STSBenchmark']
        elif self.args.eval_data == 'sick':
            best_dev = results[0]['SICKRelatedness']
            best_acc = results[1]['SICKRelatedness']
        else:
            raise ValueError(f'[Error!] Wrong evaluation data name: {self.args.eval_data}')

        return best_dev, best_acc

    def evaluate_epoch(self, mode, model_name='training'):
        print(f'Evaluating {mode}....')
        self.model.eval()
        if self.args.data_name == 'snli':
            if mode == 'Dev':
                loader = self.dev_loader
            elif mode == 'hard':
                loader = self.test_hard_loader
            else:
                loader = self.test_loader
        else:
            loader = self.test_loader
        eval_loss = 0.
        correct = 0
        representation = []
        matrix_cal = BiClassCalculator()
        alignment_values = []
        uniformity_values = []
        with torch.no_grad():
            for batch_idx, content in enumerate(loader):
                # input_info = [token_ids, [segment_ids], attention_mask]
                if self.args.do_mlm:
                    if len(content) == 7:
                        pairs_info = {'input_ids': content[0], 'attention_mask': content[1]}
                        # pairs_info = [content[0], content[1]]
                        aug_info = {'input_ids': content[2], 'attention_mask': content[3]}
                    elif len(content) == 9:
                        pairs_info = {'input_ids': content[0], 'token_type_ids': content[1],
                                      'attention_mask': content[2]}
                        # pairs_info = [content[0], content[1], content[2]]
                        aug_info = {'input_ids': content[3], 'token_type_ids': content[4], 'attention_mask': content[5]}
                    else:
                        raise Exception

                    y = content[-3]
                    pair_mlm = content[-2]
                    mlm_labels = content[-1]
                else:
                    if len(content) == 5:
                        pairs_info = {'input_ids': content[0], 'attention_mask': content[1]}
                        # pairs_info = [content[0], content[1]]
                        aug_info = {'input_ids': content[2], 'attention_mask': content[3]}
                        y = content[-1]
                    elif len(content) == 7:
                        pairs_info = {'input_ids': content[0], 'token_type_ids': content[1],
                                      'attention_mask': content[2]}
                        # pairs_info = [content[0], content[1], content[2]]
                        aug_info = {'input_ids': content[3], 'token_type_ids': content[4], 'attention_mask': content[5]}
                        y = content[-1]
                    else:
                        raise Exception

                pairs_info = {pair_key: pairs_info[pair_key].to(self.device) for pair_key in pairs_info.keys()}
                aug_info = {aug_key: aug_info[aug_key].to(self.device) for aug_key in aug_info}
                target = y.to(self.device)

                # results_info = self.model(**pairs_info)

                if self.args.aug_type == 'default':
                    results_info = self.model(
                        pair_info=pairs_info,
                        aug_pair=pairs_info,
                        pair_mlm=None
                    )
                else:
                    results_info = self.model(
                        pair_info=pairs_info,
                        aug_pair=aug_info,
                        pair_mlm=None
                    )

                if self.args.use_predict:
                    cls_emb1, cls_emb2, z1, z2, bert_outputs1, bert_outputs2, prediction = results_info
                else:
                    cls_emb1, cls_emb2, z1, z2, bert_outputs1, bert_outputs2 = results_info

                if self.args.cl_loss == 'default':
                    loss_cl = self.cl_loss(z1, z2, self.classify_loss, device=self.device)
                else:
                    loss_cl = self.cl_loss(z1, z2)

                if self.args.use_predict:
                    loss_class = self.classify_loss(prediction, target)
                    loss_cl += loss_cl + self.args.classify_weight * loss_class
                    pred = torch.max(prediction, 1)[1]
                    correct += pred.eq(target.view_as(pred)).sum().item()
                    matrix_cal.update(pred.cpu().numpy(), target.cpu().numpy())

                    for idx, item in enumerate(zip(cls_emb1, cls_emb2, pred, y)):
                        representation.append(
                            [item[0].cpu().numpy(), item[1].cpu().numpy(), item[2].cpu().item(), item[3].cpu().item()])

                eval_loss += loss_cl.item()
                # pred = torch.max(predict, 1)[1]
                # correct += pred.eq(target.view_as(pred)).sum().item()
                # matrix_cal.update(pred.cpu().numpy(), target.cpu().numpy())
                if not self.args.use_predict:
                    for idx, item in enumerate(zip(cls_emb1, cls_emb2)):
                        representation.append([item[0].cpu().numpy(), item[1].cpu().numpy()])

                if self.args.test:
                    cls_emb1_norm = F.normalize(cls_emb1, dim=1, p=2)
                    cls_emb2_norm = F.normalize(cls_emb2, dim=1, p=2)
                    uniformity_value1 = uniform_loss(x=cls_emb1_norm)
                    uniformity_value2 = uniform_loss(x=cls_emb2_norm)

                    alignment_value = align_loss(x=cls_emb1_norm, y=cls_emb2_norm)
                    alignment_values.append(alignment_value.cpu())
                    uniformity_values.append(((uniformity_value1+uniformity_value2)/2).cpu())

                    # print(f'corresponding uniformity value is: {uniformity_value1}, \t{uniformity_value2},\n'
                    #       f'corresponding alignment value is: {alignment_value}')

                if self.args.test and self.args.save_similarity:
                    t2 = 1.2
                    z1 = F.normalize(z1, dim=1)
                    z2 = F.normalize(z2, dim=1)
                    sim_mat = z1 @ z2.T

                    prob_mat = F.softmax(sim_mat * t2, 1)

                    write_csv(
                        z1=sim_mat,
                        z2=prob_mat,
                        file_name=self.args.log_path,
                        model_name=model_name
                    )

            if self.args.test:
                with open(os.path.join(self.args.log_path, f'{self.args.name}_{model_name}_{mode}_rep.pkl'), 'wb') as f:
                    pkl.dump(representation, f)

        if self.args.test:
            average_alignemnt = np.average(np.asarray(alignment_values))
            average_uniformity = np.average(np.asarray(uniformity_values))
            print(f'corresponding average alignment: {average_alignemnt}')
            print(f'corresponding average uniformity: {average_uniformity}')

        eval_loss = eval_loss / len(loader.dataset)
        if self.args.use_predict:
            return [eval_loss, matrix_cal.accuracy]
        else:
            return eval_loss

    def save_model(self, name='dev'):
        model_dict = dict()
        model_dict['state_dict'] = self.model.state_dict()
        model_dict['m_config'] = self.args
        model_dict['optimizer'] = self.optimizer.state_dict()
        if name is None:
            ckpt_path = self.ckpt_path + '.pth'
        else:
            ckpt_path = self.ckpt_path + name + '.pth'
        if not os.path.exists(os.path.dirname(ckpt_path)):
            os.makedirs(os.path.dirname(ckpt_path))
        torch.save(model_dict, ckpt_path)
        print('Saved', ckpt_path)

    def load_model(self, name='dev'):
        ckpt_path = self.ckpt_path + name + '.pth'
        print('Load checkpoint', ckpt_path)
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        try:
            self.model.load_state_dict(checkpoint['state_dict'])
        except:
            # if saving a paralleled model but loading an unparalleled model
            self.model = nn.DataParallel(self.model)
            self.model.load_state_dict(checkpoint['state_dict'])

        print(f'> best model at {ckpt_path} is loaded!')