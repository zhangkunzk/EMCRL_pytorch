"""
@Author: zhkun
@Time:  21:33
@File: model
@Description:
@Something to attention
"""
import torch
from torch import nn
from torch.cuda.amp import autocast

from transformers.models.bert.modeling_bert import BertLMPredictionHead
from transformers.models.roberta.modeling_roberta import RobertaLMHead


import photinia as ph
import os

from torch.nn.parameter import Parameter
from transformers import AutoModel, AutoConfig
from utils import ProjectionHead, ResProjectionHead, ResProjectionHeadV2, MLPLayer, Pooler, ProjectionHeadV2


class EMCL(nn.Module):
    def __init__(self, config):
        super(EMCL, self).__init__()
        self.config = config

        use_cuda = config.gpu != '' and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")

        self.bert = AutoModel.from_pretrained(config.pre_trained_model,
                                              cache_dir=config.cache_dir,
                                              output_hidden_states=True,
                                              output_attentions=False,
                                              return_dict=True,
                                              add_pooling_layer=False)
        if config.do_mlm:
            model_config = AutoConfig.from_pretrained(config.pre_trained_model, cache_dir=config.cache_dir)
            if config.pre_trained_model == 'bert-base-uncased':
                self.lm_head = BertLMPredictionHead(model_config)
            elif config.pre_trained_model == 'robera-base':
                self.lm_head = RobertaLMHead(model_config)

        if self.config.num_bert_layers == 0:
            self.num_hidden_layers = self.bert.config.num_hidden_layers
        else:
            self.num_hidden_layers = self.bert.config.num_hidden_layers if self.config.num_bert_layers > self.bert.config.num_hidden_layers else self.config.num_bert_layers

        self.pooler_type = self.config.pooler_type
        self.pooler = Pooler(self.pooler_type)

        if self.pooler_type == ['avg_first_last', 'first_last']:
            self.num_hidden_layers = self.bert.config.num_hidden_layers = 2

        # weight average
        if config.use_sentence_weight:
            self.sentence_weight = Parameter(torch.Tensor(1, self.num_hidden_layers))
            self.sentence_weight = nn.init.normal(self.sentence_weight)

        if config.use_word_attention:
            self.word_attention = ph.nn.MLPAttention(
                key_size=self.bert.config.hidden_size,
                attention_size=config.attention_size,
                use_norm=True
            )

        if self.pooler_type == 'cls':
            self.mlp_layer = MLPLayer(
                input_size=self.bert.config.hidden_size,
                output_size=self.bert.config.hidden_size
            )

        # projection
        if config.head == 'default':
            self.proj_head = ProjectionHead(self.bert.config.hidden_size, config.proj_size)
        elif config.head == 'defaultv2':
            self.proj_head = ProjectionHeadV2(self.bert.config.hidden_size, config.proj_size)
        elif config.head == 'res':
            self.proj_head = ResProjectionHead(self.bert.config.hidden_size, config.proj_size)
        elif config.head == 'resv2':
            self.proj_head = ResProjectionHeadV2(self.bert.config.hidden_size, config.proj_size)

        if config.use_predict:
            self.predict_layer = nn.Linear(in_features=self.bert.config.hidden_size, out_features=config.num_classes)

        self.dropout = nn.Dropout(p=config.dropout)

        # for display
        for param in self.bert.parameters():
            param.requires_grad = False
        if self.config.do_mlm:
            for param in self.lm_head.parameters():
                param.requires_grad = False

        self.req_grad_params = self.get_req_grad_params(debug=True)

        if config.train_bert:
            for param in self.bert.parameters():
                param.requires_grad = True
            if self.config.do_mlm:
                for param in self.lm_head.parameters():
                    param.requires_grad = True

        if self.config.use_amp and len(self.config.gpu.split(' ')) > 1:
            self.amp_mode = True
        else:
            self.amp_mode = False

    def init_linears(self):
        nn.init.uniform_(self.w_e)
        nn.init.xavier_uniform_(self.w_s.weight)
        nn.init.xavier_uniform_(self.w_t.weight)
        nn.init.xavier_uniform_(self.w_m.weight)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def single_process(self, input_ids, attention_mask, token_type_ids=None, is_train=True):
        bert_output = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

        # if len(sentence_pairs) == 3:
        #     token_ids = sentence_pairs[0]
        #     segment_ids = sentence_pairs[1]
        #     attention_masks = sentence_pairs[2]
        #     bert_output = self.bert(token_ids, token_type_ids=segment_ids,
        #                             attention_mask=attention_masks)
        # else:
        #     token_ids = sentence_pairs[0]
        #     attention_masks = sentence_pairs[1]
        #     bert_output = self.bert(token_ids, attention_mask=attention_masks)

        pool_output = self.pooler(
            outputs=bert_output,
            attention_mask=attention_mask,
            topn=self.num_hidden_layers
        )

        # size = [batch_size, length, hidden_size]
        if self.config.pooler_type in ['first_last', 'last_topn']:
            if self.config.use_sentence_weight:
                pool_output = torch.matmul(self.sentence_weight, pool_output)
                pool_output = torch.squeeze(pool_output)
            else:
                pool_output = torch.mean(pool_output, dim=2)

        # size = [batch_size, hidden_size]
        if self.config.pooler_type in ['first_last', 'last_topn']:
            if self.config.use_word_attention:
                pool_output, _ = self.word_attention(
                    key=pool_output,
                    value=pool_output,
                    key_mask=attention_mask)
            else:
                # average
                pool_output = (pool_output * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)

        if self.pooler_type == 'cls':
            pool_output = self.mlp_layer(pool_output)

        if pool_output.shape[0] > 1:
            z = self.proj_head(pool_output)
        else:
            z = pool_output

        # if self.args.use_l2_norm:
        #     z = F.normalize(z, dim=1)

        return pool_output, z, bert_output

    def process_mlm(self, mlm_input_ids, extra_info):
        # mlm_input_ids = mlm_input_ids.reshape((-1, mlm_input_ids.size(-1)))
        attention_mask = extra_info['attention_mask']
        if len(extra_info) == 2:
            token_type_ids = None
        else:
            token_type_ids = extra_info['token_type_ids']

        mlm_outputs = self.bert(
            mlm_input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )

        prediction_scores = self.lm_head(mlm_outputs.last_hidden_state)
        return prediction_scores

    def forward(self, pair_info, aug_pair=None, pair_mlm=None, is_train=True):
        with autocast(enabled=self.amp_mode):
            cls_emb1, z1, bert_outputs1 = self.single_process(**pair_info, is_train=is_train)
            if self.config.use_predict:
                predict1 = self.predict_layer(cls_emb1)
                prediction = predict1

            if aug_pair is not None:
                cls_emb2, z2, bert_outputs2 = self.single_process(**aug_pair, is_train=is_train)
                if self.config.use_predict:
                    predict2 = self.predict_layer(cls_emb2)
                    prediction = torch.mean(torch.stack([predict1, predict2], dim=0), dim=0)

            if self.config.do_mlm and pair_mlm is not None:
                mlm_prediction = self.process_mlm(pair_mlm, pair_info)

            if aug_pair is not None:
                if self.config.use_predict:
                    results = [cls_emb1, cls_emb2, z1, z2, bert_outputs1, bert_outputs2, prediction]
                else:
                    results = [cls_emb1, cls_emb2, z1, z2, bert_outputs1, bert_outputs2]

            else:
                if self.config.use_predict:
                    results = [cls_emb1, z1, bert_outputs1, prediction]
                else:
                    results = [cls_emb1, z1, bert_outputs1]

            if self.config.do_mlm and pair_mlm is not None:
                results.append(mlm_prediction)

        return results

    def get_req_grad_params(self, debug=False):
        print(f'# {self.config.name} parameters: ', end='')
        params = list()
        total_size = 0

        def multiply_iter(p_list):
            out = 1
            for _p in p_list:
                out *= _p
            return out

        for name, p in self.named_parameters():
            if p.requires_grad:
                params.append(p)
                n_params = multiply_iter(p.size())  # the product of all dimensions, i.e., # of parameters
                total_size += n_params
                if debug:
                    print(name, p.requires_grad, p.size(), multiply_iter(p.size()), sep='\t')
        print('{:,}'.format(total_size))
        return params
