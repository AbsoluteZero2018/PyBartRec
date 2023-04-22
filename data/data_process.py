# -*- coding=utf-8 -*-
# @Time: 2022/12/10 16:25
# @Author: Jeremy
# @File: data_process.py
# @Software: PyCharm
from tools.data.vocab import Vocab
from tools.model.bart import BartForClassificationAndGeneration

from typing import Union, Tuple
import logging

# --------------------------------------------------
# LOG CONFIG
# --------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)

console = logging.StreamHandler()
console.setLevel(level=logging.INFO)  # 控制台输出INFO级别以上的信息
logger.addHandler(console)

file = logging.FileHandler("data_process.log", encoding="utf-8", mode="w")
file.setLevel(level=logging.DEBUG)  # 文件输出DEBUG级别以上信息（全部信息）
formatter = logging.Formatter('[%(asctime)s | %(filename)s | line %(lineno)d] - %(levelname)s: %(message)s')
file.setFormatter(formatter)
logger.addHandler(file)

logger.debug('-' * 100)
logger.debug('Start print log')


def run_api_recommendation(
        args,
        trained_model: Union[BartForClassificationAndGeneration, str] = None,
        trained_vocab: Union[Tuple[Vocab, Vocab, Vocab], str] = None,
        only_test=False):

    logger.info('-' * 100)
    logger.info(f'进行API推荐微调')

    # --------------------------------------------------
    # datasets
    # --------------------------------------------------


def get_api_datasets(
        feature_info_path="/ecnudata/s10213903403/Python/代码补全/PyBartRec/data/datasets/encoded-data/flask_train.jsonl"):
    with open(feature_info_path) as f:
        feature_lines = f.readlines()
