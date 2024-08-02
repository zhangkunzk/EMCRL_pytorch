"""
@Author: zhkun
@Time:  21:33
@File: dataset
@Description:
@Something to attention
"""
import math
import os

import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch import nn
from typing import Optional, Union, List, Dict, Tuple

import pickle
import pandas as pd
import random
from transformers import AutoTokenizer


aug_deli = ';\t--\t;'


class SentimentDataSet(Dataset):
    def __init__(self, args, split='train'):
        self.label_dict = {}
        self.args = args
        self.split = split
        if not os.path.exists(self.args.base_path):
            self.args.base_path = '/sentence_pair'

        # self.data_path = os.path.join(self.args.base_path, self.args.data_name)
        self.pkl_path = os.path.join(self.args.base_path, self.args.data_name, 'pkl_data')

        if self.args.data_name == 'quora':
            self.label_dict = {'0': 0, '1': 1}
        else:
            self.label_dict = {'entailment': 0, 'neutral': 1, 'contradiction': 2}

        if self.split == 'train':
            self.data_path = os.path.join(self.args.base_path, self.args.data_name,
                                          self.args.data_name +'_train_aug.csv')
        elif self.split == 'dev':
            self.data_path = os.path.join(self.args.base_path, self.args.data_name,
                                          self.args.data_name +'_dev_aug.csv')
        elif self.split == 'test':
            self.data_path = os.path.join(self.args.base_path, self.args.data_name,
                                          self.args.data_name + '_test_aug.csv')
        elif self.split == 'hard':
            self.data_path = os.path.join(self.args.base_path, self.args.data_name,
                                          self.args.data_name + '_test_hard_aug.csv')
        else:
            raise ValueError('wrong file name, please check again')

        # self.label_desp_path = os.path.join(self.args.base_path, self.args.data_name,
        #                                     f'{self.args.data_name}_label_desp.txt')

        lines = open(self.data_path, 'r', encoding='utf8')
        self.total_length = sum(1 for i in lines) - 1

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.args.pre_trained_model,
            do_lower_case=True,
            cache_dir=self.args.cache_dir
        )

        self.origin_pddata = pd.read_csv(self.data_path, ';\t;')
        # self.origin_pddata = self.origin_pddata[["label", "Qtitle", "Answer"]]

        # self.load_desp()


    def __getitem__(self, index):

        current_sample = self.origin_pddata.loc[index]
        sentence1 = current_sample['sentence1']
        sentence2 = current_sample['sentence2']
        sentence1_aug = self.obtain_aug(current_sample['sentence1_aug'])
        sentence2_aug = self.obtain_aug(current_sample['sentence2_aug'])
        label_index = self.label_dict[current_sample['label']]

        try:
            values = self.process_sent(sent1=sentence1, sent2=sentence2)
            values_aug = self.process_sent(sent1=sentence1_aug, sent2=sentence2_aug)

            if len(values) == 2:
                return values[0], values[1], values_aug[0], values_aug[1], label_index
            else:
                return values[0], values[1], values[2], values_aug[0], values_aug[1], values_aug[2], label_index

        except TypeError:
            print(sentence1)
            print(sentence2)
            print(sentence1_aug)
            print(sentence2_aug)

    def __len__(self):
        return self.total_length

    def obtain_aug(self, content):
        try:
            sentences = content.split(aug_deli)
            count = len(sentences)
            selected_id = random.randint(0, count-1)
            # print(f'> current select idx for augment sentence is {selected_id}')
            result = sentences[selected_id].strip()
        except AttributeError:
            print(content)
            exit(-1)

        return result

    def process_sent(self, sent1, sent2=None):
        if sent2 is not None:
            results = self.tokenizer.encode_plus([sent1, sent2])
        else:
            results = self.tokenizer.encode_plus(sent1)

        sentence_ids = torch.tensor(results['input_ids'])
        attention_mask_ids = torch.tensor(results['attention_mask'])
        if 'token_type_ids' in results.keys():
            segment_ids = torch.tensor(results['token_type_ids'])
            values = [sentence_ids, segment_ids, attention_mask_ids]
        else:
            values = [sentence_ids, attention_mask_ids]

        return values


class SentimentData(Dataset):
    def __init__(self, args):
        self._args = args
        if self._args.debug:
            self._train_set = SentimentDataSet(self._args, split='dev')
        else:
            self._train_set = SentimentDataSet(self._args, split='train')

        self._test_set = SentimentDataSet(self._args, split='test')
        self._dev_set = SentimentDataSet(self._args, split='dev')
        if args.data_name == 'snli':
            self._hard_set = SentimentDataSet(self._args, split='hard')

        if args.do_mlm:
            self.tokenizer = AutoTokenizer.from_pretrained(
                args.pre_trained_model,
                do_lower_case=True,
                cache_dir=args.cache_dir
            )


    def get_dataloaders(self, batch_size, shuffle=True, num_workers=4, pin_memory=True):
        train_loader = self.get_loader(
            batch_size=batch_size,
            type='train',
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory
        )

        dev_loader = self.get_loader(
            batch_size=batch_size,
            type='dev',
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )

        test_loader = self.get_loader(
            batch_size=batch_size,
            type='test',
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )

        if self._args.data_name == 'snli':
            hard_loader = self.get_loader(
                batch_size=batch_size,
                type='hard',
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory
            )
            return train_loader, dev_loader, test_loader, hard_loader
        else:
            return train_loader, dev_loader, test_loader

    def get_loader(self, batch_size, type='train', shuffle=True, num_workers=4, pin_memory=True):
        if type == 'train':
            current_dataset = self._train_set
        elif type == 'dev':
            current_dataset = self._dev_set
        elif type == 'test':
            current_dataset = self._test_set
        elif type == 'hard':
            current_dataset = self._hard_set
        else:
            raise ValueError

        loader = DataLoader(
            current_dataset,
            shuffle=shuffle,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=self._custom_fn,
            pin_memory=pin_memory,
            drop_last=True
        )

        return loader

    def _custom_fn(self, batch):
        # print(len(batch))
        # print(len(batch[0]))

        input_text = []
        batch_count = len(batch[0])
        for idx in range(batch_count - 1):
            if idx == 1:
                input_text.append(pad_sequence([item[idx] for item in batch], batch_first=True, padding_value=1))
            elif batch_count == 5 and idx == 3:
                input_text.append(pad_sequence([item[idx] for item in batch], batch_first=True, padding_value=1))
            elif batch_count == 7 and idx == 4:
                input_text.append(pad_sequence([item[idx] for item in batch], batch_first=True, padding_value=1))
            else:
                input_text.append(pad_sequence([item[idx] for item in batch], batch_first=True))

        labels = torch.tensor([item[-1] for item in batch])

        if self._args.do_mlm:
            input_replace, replace_labels = self.mask_tokens(input_text[0])
            if len(input_text) == 4:
                return input_text[0], input_text[1], input_text[2], input_text[3], labels, input_replace, replace_labels
            else:
                return input_text[0], input_text[1], input_text[2], input_text[3], input_text[4], input_text[5], labels, input_replace, replace_labels
        else:
            if len(input_text) == 4:
                return input_text[0], input_text[1], input_text[2], input_text[3], labels
            else:
                return input_text[0], input_text[1], input_text[2], input_text[3], input_text[4], input_text[5], labels

    def mask_tokens(
            self,
            inputs: torch.Tensor,
            special_tokens_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self._args.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels


class WikiDataSet(Dataset):
    def __init__(self, args, split='train'):
        self.label_dict = {}
        self.args = args
        self.split = split
        # if not os.path.exists(self.args.base_path):
        self.args.base_path = '/SentEval'
        if not os.path.exists(self.args.base_path):
            self.args.base_path = 'D:/data/SentEval'

        # self.data_path = self.args.base_path, self.args.data_name
        self.pkl_path = os.path.join(self.args.base_path,  self.args.data_name, 'pkl_data')

        self.label_dict = {'0': 0, '1': 1}

        if self.split =='train':
            self.data_path = os.path.join(self.args.base_path, self.args.data_name, 'wiki_train_aug.csv')
        else:
            self.data_path = os.path.join(self.args.base_path, self.args.data_name, 'wiki_test_aug.csv')

        lines = open(self.data_path, 'r', encoding='utf8')
        self.total_length = sum(1 for i in lines) - 1

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.args.pre_trained_model,
            do_lower_case=True,
            cache_dir=self.args.cache_dir
        )

        self.origin_pddata = pd.read_csv(self.data_path, ';\t;')

    def __getitem__(self, index):
        current_sample = self.origin_pddata.loc[index]
        sentence1 = current_sample['sentence1']
        sentence1_aug = self.obtain_aug(current_sample['sentence1_aug'])
        label_index = self.label_dict[str(random.randint(0, 1))]

        try:
            values = self.process_sent(sent1=sentence1)
            values_aug = self.process_sent(sent1=sentence1_aug)

            if len(values) == 2:
                return values[0], values[1], values_aug[0], values_aug[1], label_index
            else:
                return values[0], values[1], values[2], values_aug[0], values_aug[1], values_aug[2], label_index

        except TypeError:
            print(sentence1)
            print(sentence1_aug)

    def __len__(self):
        return self.total_length

    def obtain_aug(self, content):
        try:
            sentences = content.strip().split(aug_deli)
            num_aug = len(sentences)
            selected_id = random.randint(0, num_aug-1)
            result = sentences[selected_id]
        except AttributeError:
            print(content)
            exit(-1)

        return result

    def process_sent(self, sent1, sent2=None):
        if sent2 is not None:
            results = self.tokenizer.encode_plus([sent1, sent2])
        else:
            results = self.tokenizer.encode_plus(sent1)

        sentence_ids = torch.tensor(results['input_ids'])
        attention_mask_ids = torch.tensor(results['attention_mask'])
        if 'token_type_ids' in results.keys():
            segment_ids = torch.tensor(results['token_type_ids'])
            values = [sentence_ids, segment_ids, attention_mask_ids]
        else:
            values = [sentence_ids, attention_mask_ids]

        return values


class WikiData(Dataset):
    def __init__(self, args):
        self._args = args
        if self._args.debug:
            self._train_set = WikiDataSet(self._args, split='test')
        else:
            self._train_set = WikiDataSet(self._args, split='train')

        self._test_set = WikiDataSet(self._args, split='test')

        if args.do_mlm:
            self.tokenizer = AutoTokenizer.from_pretrained(
                args.pre_trained_model,
                do_lower_case=True,
                cache_dir= args.cache_dir
            )

    def get_dataloaders(self, batch_size, shuffle=True, num_workers=4, pin_memory=True):
        train_loader = self.get_loader(
            batch_size=batch_size,
            type='train',
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory
        )

        test_loader = self.get_loader(
            batch_size=batch_size,
            type='test',
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )

        return train_loader, test_loader

    def get_loader(self, batch_size, type='train', shuffle=True, num_workers=4, pin_memory=True):
        if type == 'train':
            current_dataset = self._train_set
        elif type == 'test':
            current_dataset = self._test_set
        else:
            raise ValueError

        loader = DataLoader(
            current_dataset,
            shuffle=shuffle,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=self._custom_fn,
            pin_memory=pin_memory,
            drop_last=True
        )

        return loader

    def _custom_fn(self, batch):
        # print(len(batch))
        # print(len(batch[0]))

        input_text = []
        batch_count = len(batch[0])
        for idx in range(batch_count - 1):
            if idx == 1:
                input_text.append(pad_sequence([item[idx] for item in batch], batch_first=True, padding_value=1))
            elif batch_count == 5 and idx == 3:
                input_text.append(pad_sequence([item[idx] for item in batch], batch_first=True, padding_value=1))
            elif batch_count == 7 and idx == 4:
                input_text.append(pad_sequence([item[idx] for item in batch], batch_first=True, padding_value=1))
            else:
                input_text.append(pad_sequence([item[idx] for item in batch], batch_first=True))

        labels = torch.tensor([item[-1] for item in batch])

        if self._args.do_mlm:
            input_replace, replace_labels = self.mask_tokens(input_text[0])
            if len(input_text) == 4:
                return input_text[0], input_text[1], input_text[2], input_text[3], labels, input_replace, replace_labels
            else:
                return input_text[0], input_text[1], input_text[2], input_text[3], input_text[4], input_text[5], labels, input_replace, replace_labels
        else:
            if len(input_text) == 4:
                return input_text[0], input_text[1], input_text[2], input_text[3], labels
            else:
                return input_text[0], input_text[1], input_text[2], input_text[3], input_text[4], input_text[5], labels

    def mask_tokens(
            self,
            inputs: torch.Tensor,
            special_tokens_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self._args.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels
