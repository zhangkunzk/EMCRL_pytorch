"""
@Author: zhkun
@Time:  21:34
@File: my_parser
@Description: parameter settings
@Something to attention
"""
import argparse
import pprint


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default="emcl")

    parser.add_argument('--net', type=str, default='emcl')
    parser.add_argument('--data_name', type=str, default='wiki')
    parser.add_argument('--base_path', type=str, default='/data/sentence_pair')
    parser.add_argument('--log_path', type=str, default='logs')
    parser.add_argument('--cache_dir', type=str, default='/data/pretrained_models')

    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--noise_prob', type=float, default=1e-4)

    parser.add_argument('--num_bert_layers', type=int, default=2)
    parser.add_argument('--in_features', type=int, default=768)
    parser.add_argument('--attention_size', type=int, default=100)
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--mlp_size', type=int, default=200)
    parser.add_argument('--proj_size', type=int, default=300)

    parser.add_argument('--display_step', type=int, default=50)
    parser.add_argument('--weight_decay', type=float, default=0.005, help='add l2-norm to the added layers except Bert')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--accumulate_step', type=int, default=20,
                        help='leverage additional step power to extend the batch size')
    parser.add_argument('--epochs', type=int, default=6)
    parser.add_argument('--test', action='store_true', default=False, help='Whether to just test the model')
    parser.add_argument('--pre_trained_model', type=str, default='bert-base-uncased',
                        help='bert-base-uncased, bert-large-uncased')

    parser.add_argument('--classify_weight', type=float, default=0.1, help='add prediction target for model learning')
    parser.add_argument('--mlm_probability', type=float, default=0.1, help='add prediction target for model learning')
    parser.add_argument('--mlm_weight', type=float, default=0.5, help='the weight for the mlm task loss')
    parser.add_argument('--cl_tempure', type=float, default=0.06, help='the tempure to control the cl loss')

    parser.add_argument('--gpu', type=str, default='0', help='which gpus to use')
    parser.add_argument('--train_bert', action='store_true', default=False, help='Whether to fine-tune bert')

    # different selection information
    parser.add_argument('--debug', action='store_true', default=False,
                        help='whether debug the whole model, [true, false]')
    parser.add_argument('--use_predict', action='store_true', default=True,
                        help='whether use prediction layer for downstream tasks')
    parser.add_argument('--use_sentence_weight', action='store_true', default=False,
                        help='whether use sentence weight to get the value')
    parser.add_argument('--use_origin', action='store_true', default=False,
                        help='whether use sentence weight to get the value')
    parser.add_argument('--use_word_attention', action='store_true', default=False,
                        help='whether use sentence weight to get the value')
    parser.add_argument('--do_mlm', action='store_true', default=False,
                        help='whether use mlm as the extra task')
    parser.add_argument('--use_amp', action='store_true', default=False,
                        help='whether use mlm as the extra task')
    parser.add_argument('--save_similarity', action='store_true', default=False,
                        help='whether save the similarity results for model evaluation')
    parser.add_argument('--use_evaluation', action='store_true', default=False,
                        help='whether evaluation the model using the SentEval toolkit')

    parser.add_argument('--align_uniform', action='store_true', default=False,
                        help='whether use the alignment and uniformity as additional loss func')
    parser.add_argument('--align_loss_type', type=str, default='all',
                        help='different alignment and uniformtiy loss function, [all, align, uniform]')

    # different type information
    parser.add_argument('--pooler_type', type=str, default='avg_first_last',
                        help='which pooler type is used {cls, cls_before_pooler, avg, avg_topn,avg_first_last, first_last, last_topn}')
    parser.add_argument('--head', type=str, default='default',
                        help='which head type is used for cl{default, res, resV2}')
    parser.add_argument('--cl_loss', type=str, default='defaule',
                        help='which head type is used for cl{default, only_true, em}')
    parser.add_argument('--aug_type', type=str, default='default',
                        help='which augmentation type is used{default, eda}')
    parser.add_argument('--eval_data', type=str, default='stsb',
                        help='which data is used as dev set for wiki training {stsb, sick}')

    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lr_decay', type=float, default=0.95)
    parser.add_argument('--grad_max_norm', type=float, default=0.)  #
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--num_pred_layer', type=int, default=2)

    args = parser.parse_args()
    # if args.debug:
    pprint.PrettyPrinter().pprint(args.__dict__)
    return args