"""
@Author: zhkun
@Time:  11:26
@File: evaluate_case
@Description: 直接输出得到结果的热力图和相似度值
@Something to attention
"""
from data.dataset import SentimentData, WikiData
import torch
import os
import logging
from tqdm import tqdm
import numpy as np
from numpy import ndarray
import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor, device
import pandas as pd
import seaborn as sns
if os.name != 'nt':
    import matplotlib as mpl
    mpl.use('Agg')
import matplotlib.pyplot as plt


from my_parser import parser
from model import EMCL
from utils import write_file

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model(model, path, device):
    ckpt_path = path
    print('Load checkpoint', ckpt_path)
    checkpoint = torch.load(ckpt_path, map_location=device)
    try:
        model.load_state_dict(checkpoint['state_dict'])
        # 因为训练代码中有nn.DataParallel()，导致模型是按照并行模式训练的，把这部分去掉就好了，或者直接load并行的
        # checkpoint = {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}
        # model.load_state_dict(checkpoint)
    except:
        # if saving a paralleled model but loading an unparalleled model
        model = nn.DataParallel(model)
        model.load_state_dict(checkpoint['state_dict'])

    print(f'> model at {ckpt_path} is loaded!')


def write_excel(z1, z2, file_name, model_name=None):
    z1 = z1.cpu().numpy()
    z2 = z2.cpu().numpy()

    z1 = pd.DataFrame(z1)
    z2 = pd.DataFrame(z2)


    if model_name is not None:
        file_name = file_name + '/' + model_name

    writer1 = pd.ExcelWriter(file_name+'_sim_mat.xlsx', mode='w')

    z1.to_excel(
        excel_writer=writer1,
        sheet_name='Sheet1',
        float_format='%.5f'
    )
    z2.to_excel(
        excel_writer=writer1,
        sheet_name='Sheet2',
        float_format='%.5f'
    )

    writer1.save()


def load_excel(file_name, sheet='Sheet1'):
    df_results = pd.read_excel(
        io=file_name,
        sheet_name=sheet,
    )

    return df_results


def write_csv(z1, z2, file_name, model_name=None):
    z1 = pd.DataFrame(z1)
    z2 = pd.DataFrame(z2)

    if model_name is not None:
        file_name = file_name + '/' + model_name

    z1.to_csv(file_name+'_sim_mat.csv', float_format='%.5f', sep=',', mode='a', index=False, header=False)
    z2.to_csv(file_name + '_softmax_mat.csv', float_format='%.5f', sep=',', mode='a', index=False, header=False)


def load_csv(file_name):

    df_data = pd.read_csv(file_name, sep=',')

    return df_data


def draw_heatmap(values, max_value=1, min_value=-1):
    plt.subplots(figsize=(15, 10))  # 设置画面大小
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体

    sns.set(font_scale=1.0)
    hm = sns.heatmap(values, vmin=min_value, vmax=max_value, cmap='RdBu_r')

    plt.show()


def draw_figure(file_name, my_args=None, save_model_type=None):
    if my_args is None:
        my_args = parser()

    if save_model_type is not None:
        model_name = my_args + save_model_type
    else:
        model_name = 'wiki_flddtest_loss'

    file_name = os.path.join(my_args.log_path, model_name, file_name)
    results = load_csv(file_name)

    length = len(results)

    size = int(length // my_args.batch_size)

    for idx in range(size):
        start = idx * my_args.batch_size
        end = start + my_args.batch_size
        current_data = results.values[start:end]

        min_value = current_data.min()
        max_value = 1

        if min_value < 0:
            min_value = -1
        else:
            min_value = 0

        print(f'> starting drawing data from {start}-{end}')
        draw_heatmap(current_data, max_value=max_value, min_value=min_value)


def obtain_result(my_args=None, save_model_type=None):
    if my_args is None:
        my_args = parser()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if save_model_type is not None:
        model_name = my_args + save_model_type
    else:
        model_name = 'wiki_flddtest_loss'

    model = EMCL(my_args)
    model.to(device)
    ckpt_path = os.path.join('checkpoint', '{}.pth'.format(model_name))
    load_model(model, ckpt_path, device)

    if my_args.data_name.lower() in ['sick', 'scitail', 'quora', 'msrp', 'snli']:
        dataset = SentimentData(my_args)
    elif my_args.data_name.lower() == 'wiki':
        dataset = WikiData(my_args)
    else:
        raise ValueError(f'wrong dataset name {my_args.data_name.lower()}, please try again')

    test_loader = dataset.get_loader(
        type='test',
        batch_size=my_args.batch_size,
        num_workers=1,      # 保证顺序读取
        pin_memory=device == 'cuda'
    )

    with torch.no_grad():
        for batch_idx, content in enumerate(test_loader):
            # input_info = [token_ids, [segment_ids], attention_mask]
            if my_args.do_mlm:
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

            pairs_info = {pair_key: pairs_info[pair_key].to(device) for pair_key in pairs_info.keys()}
            aug_info = {aug_key: aug_info[aug_key].to(device) for aug_key in aug_info}
            target = y.to(device)

            # results_info = self.model(**pairs_info)

            if my_args.aug_type == 'default':
                results_info = model(
                    pair_info=pairs_info,
                    aug_pair=pairs_info,
                    pair_mlm=None
                )
            else:
                results_info = model(
                    pair_info=pairs_info,
                    aug_pair=aug_info,
                    pair_mlm=None
                )

            if my_args.use_predict:
                cls_emb1, cls_emb2, z1, z2, bert_outputs1, bert_outputs2, prediction = results_info
            else:
                cls_emb1, cls_emb2, z1, z2, bert_outputs1, bert_outputs2 = results_info

            #
            t2 = 1.2
            z1 = F.normalize(z1, dim=1)
            z2 = F.normalize(z2, dim=1)
            sim_mat = z1 @ z2.T

            prob_mat = F.softmax(sim_mat * t2, 1)

            write_csv(
                z1=sim_mat,
                z2=prob_mat,
                file_name=my_args.log_path,
                model_name=model_name
            )







