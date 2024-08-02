"""
@Author: zhkun
@Time:  17:27
@File: main
@Description: main entrance
@Something to attention
"""

import warnings

warnings.filterwarnings(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings("ignore")

import os
from my_parser import parser
args = parser()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
os.environ['TOKENIZERS_PARALLELISM'] = "false"
# 控制pytorch从本地读入
os.environ['HF_DATASETS_OFFLINE'] = str(1)
os.environ['TRANSFORMERS_OFFLINE'] = str(1)

from solver import Solver
import pretty_errors


def main():
    # args = parser()
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if args.net in ['emcl', 'EMCL']:
        solver = Solver(args)
    else:
        raise ValueError('the key word is not exist')
    if not args.test:
        solver.train()
    else:
        solver.test()


if __name__ == '__main__':
    main()
