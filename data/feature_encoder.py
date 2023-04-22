# -*- coding=utf-8 -*-
# @Time: 2022/12/31 15:30
# @Author: Jeremy
# @File: feature_encoder.py
# @Software: PyCharm
import re
import os
import json
import string
import logging
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from typing import Tuple, Any, Dict, Generator, List
from config import config
from nltk.tokenize import word_tokenize
from tools import get_dataflow
from data_utils import DataUtils
from tools.path_utils import *
from tools.asts.ast_parser import generate_single_ast_nl
from tools.data_utils import replace_string_literal

# --------------------------------------------------
# LOG CONFIG
# --------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)

console = logging.StreamHandler()
console.setLevel(level=logging.INFO)  # 控制台输出INFO级别以上的信息
logger.addHandler(console)

file = logging.FileHandler(config["log"]["feature_encoder"], encoding="utf-8", mode="w")
file.setLevel(level=logging.DEBUG)  # 文件输出DEBUG级别以上信息（全部信息）
formatter = logging.Formatter('[%(asctime)s | %(filename)s | line %(lineno)d] - %(levelname)s: %(message)s')
file.setFormatter(formatter)
logger.addHandler(file)

logger.debug('-' * 100)
logger.debug('Start print log')

if os.cpu_count() is None:
    processes_num = 0
else:
    processes_num = os.cpu_count() - 1
if processes_num > 15:
    processes_num = 15


class FeatureEncoder:
    def __init__(self, proj_name=config["project"]["proj_name"], split="train"):
        # , datasets_path="datasets", clean_data_dir="clean-data", json_data_dir="json-data",
        #              encoded_data_dir="encoded-data", feature_info_file="flask.train.jsonl"
        # self.json_data_path = os.path.join(datasets_path, json_data_dir)  # TODO:这里的文件夹可能不存在，如果不存在需要新建
        # self.encoded_data_path = os.path.join(datasets_path, encoded_data_dir)
        # self.clean_data_path = os.path.join(datasets_path, clean_data_dir)
        self.feature_info_path = config["project"]["proj_train_json_list"].format(proj_name)
        self.feature_info_path_list = config["project"]["proj_train_json_list_files"].format(proj_name)  # 存放训练集文件列表的文件
        # self.raw_data_path = RawFilePath()
        # self.clean_data_path = CleanFilePath()
        # self.json_data_path = JsonFilePath(proj_name)
        # self.encoded_data_path = EncodedFilePath()

        self.proj_name = proj_name
        self.split = split

        self.proj_depends, self.proj_token_count, self.proj_token_no = self.get_proj_info()

    def get_proj_info(self):
        if os.path.isfile(config["project"]["depends"].format(self.proj_name)) \
                and os.path.isfile(config["project"]["token_count"].format(self.proj_name)) \
                and os.path.isfile(config["project"]["token_no"].format(self.proj_name)):
            with open(config["project"]["depends"].format(self.proj_name)) as f:
                proj_depends = json.load(f)
            with open(config["project"]["token_count"].format(self.proj_name)) as f:
                proj_token_count = json.load(f)
            with open(config["project"]["token_no"].format(self.proj_name)) as f:
                proj_token_no = json.load(f)
        else:
            proj_depends, proj_token_count, proj_token_no = self.get_proj_tokens()
        # for json_file_path in self.json_data_path.glob("*.json"):
        #     # json_file_path = os.path.join(self.json_data_path, json_file)
        #     with open(json_file_path) as f:
        #         if json_file == "proj_depends.json":  # TODO:这个文件可能不存在，以下两个文件同理
        #             proj_depends = json.load(f)
        #         elif json_file == "proj_token_count.json":
        #             proj_token_count = json.load(f)
        #         else:
        #             proj_token_no = json.load(f)
        return proj_depends, proj_token_count, proj_token_no

    def get_proj_tokens(self):

        ret_proj_depends = {}
        ret_proj_token_count = {}
        ret_proj_token_no = {}
        clean_file_list = DataUtils.get_file_list(CleanFilePath(self.proj_name))

        # TODO:如何处理下面三行代码，以便其他方法能够复用
        del_estr = string.punctuation + string.digits
        replace = " " * len(del_estr)
        tran_tab = str.maketrans(del_estr, replace)

        for file in clean_file_list:
            logger = file.get_logger()
            with open(file, encoding='utf-8') as f:
                lines = f.readlines()
            file = str(file)

            logger.info('-' * 100)
            logger.info(f"正在获取{file}的token信息")

            line_label = 0
            for i in range(0, len(lines)):

                line_label += 1
                if lines[i].strip() == '':
                    continue
                elif re.sub(' ', '', lines[i].strip()) == '':
                    continue
                elif 'import ' in lines[i]:
                    # 前面已经存放过当前文件的依赖项，因此取出，用于更新，主要是统计import那一行的形式
                    if file in ret_proj_depends:
                        imports = ret_proj_depends[file]
                    else:
                        imports = []
                    # 将import那行加入当前文件的依赖项
                    imports.append(lines[i].strip())
                    ret_proj_depends[file] = imports

                tmp = lines[i].strip().translate(tran_tab)
                tokens = word_tokenize(tmp)

                # logger.debug("分词标记：" + json.dumps(tokens))

                # 每个文件token的数量
                for tk in tokens:
                    token = f"{tk}##{file}"
                    # token的数量
                    if token in ret_proj_token_count:
                        token_count = ret_proj_token_count[token]
                    else:
                        token_count = 0
                    token_count += lines[i].count(tk)
                    ret_proj_token_count[token] = token_count
                    # token所在的行号
                    if token in ret_proj_token_no:
                        no = ret_proj_token_no[token]
                    else:
                        no = []
                    no.append(line_label)
                    ret_proj_token_no[token] = no

        self.save_proj_tokens(ret_proj_depends, ret_proj_token_count, ret_proj_token_no)

        return ret_proj_depends, ret_proj_token_count, ret_proj_token_no

    def save_proj_tokens(self, proj_depends, proj_token_count, proj_token_no):
        # if not os.path.exists(self.json_data_path): # 判断不存在的应放到外层
        #     os.makedirs(self.json_data_path)
        with open(config["project"]["token_count"].format(self.proj_name), "w") as f:
            json.dump(proj_token_count, f, indent=4, ensure_ascii=False)
        with open(config["project"]["depends"].format(self.proj_name), "w") as f:
            json.dump(proj_depends, f, indent=4, ensure_ascii=False)
        with open(config["project"]["token_no"].format(self.proj_name), "w") as f:
            json.dump(proj_token_no, f, indent=4, ensure_ascii=False)

    def encode_features(self):
        # if not os.path.exists(self.encoded_data_path):
        #     os.mkdir(self.encoded_data_path)

        # feature_data = ""
        # feature_label = ""

        feature_data_path = config["project"]["proj_feature_data"].format(self.proj_name)
        feature_label_path = config["project"]["proj_feature_label"].format(self.proj_name)

        all_pyart_input = {
            "all_dataflow_scores": [],
            "all_tosim_scores": [],
            "all_line_scores": [],
            "all_conum_scores": []
        }
        all_spt_code_input = {
            "all_codes": [],
            "all_asts": [],
            "all_nls": []
        }
        all_labels = []
        all_rec_info = []

        with open(self.feature_info_path_list) as feature_info_list_file:
            feature_info_list_files = feature_info_list_file.read().splitlines()
        # with open(self.feature_info_path) as feature_info_file:
        #     feature_info_list = feature_info_file.readlines()
        # 直接迭代file对象应该是线程不安全的
        # 参考：https://www.jianshu.com/p/d1e05a3e32c8
        # 参考：https://www.zhihu.com/question/41493561
        if processes_num > 0:
            with ProcessPoolExecutor(processes_num) as executor:
                for current_labels, test_info, spt_code_input, pyart_input in executor.map(
                        self.single_process_encode_features, feature_info_list_files[:3]):
                    all_pyart_input["all_dataflow_scores"].extend(pyart_input["current_dataflow_scores"])
                    all_pyart_input["all_tosim_scores"].extend(pyart_input["current_tosim_scores"])
                    all_pyart_input["all_line_scores"].extend(pyart_input["current_line_scores"])
                    all_pyart_input["all_conum_scores"].extend(pyart_input["current_conum_scores"])
                    all_spt_code_input["all_codes"].extend(spt_code_input["current_codes"])
                    all_spt_code_input["all_asts"].extend(spt_code_input["current_asts"])
                    all_spt_code_input["all_nls"].extend(spt_code_input["current_nls"])
                    all_labels.extend(current_labels)
                    all_rec_info.extend(test_info)
        else:
            for current_labels, test_info, spt_code_input, pyart_input in map(
                    self.single_process_encode_features, feature_info_list_files[:3]):
                all_pyart_input["all_dataflow_scores"].extend(pyart_input["current_dataflow_scores"])
                all_pyart_input["all_tosim_scores"].extend(pyart_input["current_tosim_scores"])
                all_pyart_input["all_line_scores"].extend(pyart_input["current_line_scores"])
                all_pyart_input["all_conum_scores"].extend(pyart_input["current_conum_scores"])
                all_spt_code_input["all_codes"].extend(spt_code_input["current_codes"])
                all_spt_code_input["all_asts"].extend(spt_code_input["current_asts"])
                all_spt_code_input["all_nls"].extend(spt_code_input["current_nls"])
                all_labels.extend(current_labels)
                all_rec_info.extend(test_info)

        pyart_data = pd.DataFrame(all_pyart_input)
        pyart_data.to_csv(feature_data_path)
        pyart_label = pd.DataFrame({"label": all_labels})
        pyart_label.to_csv(feature_label_path)

        assert self.split in ["train", "valid", "split"]
        if self.split == "train":
            return all_pyart_input, all_spt_code_input, all_labels
        elif self.split in ["valid", "test"]:
            return all_pyart_input, all_spt_code_input, all_labels, all_rec_info

    def single_process_encode_features(self, rec_feature_file: str) -> Tuple[List[int], List[Dict[str, Any]], Dict,
                                                                             Dict]:
        collected_candidate_apis = []
        test_info = []
        current_dataflow_scores = []
        current_tosim_scores = []
        current_line_scores = []
        current_conum_scores = []
        current_codes = []
        current_asts = []
        current_nls = []
        current_labels = []

        with open(rec_feature_file) as f:
            feature_info_list = f.readlines()
        for feature_info_str in feature_info_list:
            feature_info = json.loads(feature_info_str)
            rec_point = feature_info["rec_point"]
            naming_line = feature_info["naming_line"]
            naming_context = feature_info["naming_context"]
            candidate_apis = feature_info["candidate_apis"]
            file_path = CleanFilePath(feature_info["file"])
            positive_api = feature_info["positive_api"]
            caller_type = feature_info["caller_type"]
            current_data_flow = feature_info["current_data_flow"]
            max_flow = feature_info["max_flow"]
            code_tokens = feature_info["code_tokens"]
            context_for_ast = feature_info["context_for_ast"]

            code = replace_string_literal(' '.join(code_tokens))
            ast, nl = generate_single_ast_nl(source=context_for_ast, lang="python")

            dataflow_scores = get_dataflow.get_dataflow_scores(candidate_apis, max_flow, current_data_flow,
                                                               caller_type, positive_api)
            tosim_scores = get_dataflow.get_tosim_scores(candidate_apis, max_flow, current_data_flow, caller_type,
                                                         positive_api)
            line_scores = self.get_line_scores(candidate_apis, naming_line, naming_context, file_path)

            if caller_type == 'None' or caller_type == 'Any' or caller_type == 'nothing':
                for api in candidate_apis:
                    if api.startswith('__') or re.match('[A-Z0-9_]+$', api) or api.strip() == '_':
                        file_path.get_logger().debug(f"当前候选API{api}有问题，跳过此API")
                        continue
                    if api == positive_api:
                        label = 1
                    else:
                        label = 0

                    current_dataflow_scores.append(dataflow_scores[api])
                    current_tosim_scores.append(tosim_scores[api])
                    current_line_scores.append(line_scores[api])
                    current_conum_scores.append(0.0)
                    current_labels.append(label)
                    collected_candidate_apis.append(api)
                    current_codes.append(code)
                    current_asts.append(ast)
                    current_nls.append(nl)
            else:
                conum_scores = self.get_conum_scores(candidate_apis, naming_context, file_path)
                for api in candidate_apis:
                    if api.startswith('__') or re.match('[A-Z0-9_]+$', api) or api.strip() == '_':
                        continue
                    if api == positive_api:
                        label = 1
                    else:
                        label = 0

                    current_dataflow_scores.append(dataflow_scores[api])
                    current_tosim_scores.append(tosim_scores[api])
                    current_line_scores.append(line_scores[api])
                    current_conum_scores.append(conum_scores[api])
                    current_labels.append(label)
                    collected_candidate_apis.append(api)
                    current_codes.append(code)
                    current_asts.append(ast)
                    current_nls.append(nl)

            test_info.append({
                "rec_point": rec_point,
                "collected_apis": collected_candidate_apis,
                "positive_api": positive_api
            })

        pyart_input = {
            "current_dataflow_scores": current_dataflow_scores,
            "current_tosim_scores": current_tosim_scores,
            "current_line_scores": current_line_scores,
            "current_conum_scores": current_conum_scores
        }
        spt_code_input = {
            "current_codes": current_codes,
            "current_asts": current_asts,
            "current_nls": current_nls
        }

        return current_labels, test_info, spt_code_input, pyart_input

    # 所有候选api和推荐点之前的上下文同一行的共现分数
    def get_line_scores(self, candidate_apis, naming_line, naming_context, file_path: CleanFilePath):
        line_scores = {}
        fi = re.sub(r'\.py', '', str(file_path))
        index = fi.rfind('/')

        cur_name = fi[index + 1:]  # 当前文件名（无后缀）
        files = []
        # 看k引入的依赖中有无当前文件，有则不统计这个k（会重复统计）
        for k, v in self.proj_depends.items():
            k = CleanFilePath(k)
            if k == file_path:
                continue
            flag = 0
            for imports in v:
                # 如果项目中其他文件导入了当前文件，就不加入files
                if cur_name in imports:
                    flag = 1
                    break
            if flag == 0:
                files.append(k)
        file_path.get_logger().debug(f"{file_path}中计算候选API和上下文token在其他{files}文件中和当前文件中的共现分数")
        for api in candidate_apis:
            # 只寻找格式正确的api
            if api.startswith('__') or re.match('[A-Z0-9_]+$', api) or api.strip() == '_':
                continue
            line_ret = self.get_conum_of_line(api, naming_line, naming_context, files)
            line_scores[api] = line_ret
        return line_scores

    # 同一行api和token的共现分数
    def get_conum_of_line(self, api, naming_line, naming_context, files: List[CleanFilePath]):

        del_estr = string.punctuation + string.digits  # 所有的标点符号和数字
        replace = " " * len(del_estr)  # 重复空格多少次
        tran_tab = str.maketrans(del_estr, replace)  # 翻译表，将所有的标点符号和数字替换为空格

        tmp = naming_line.translate(tran_tab)
        cs = api.translate(tran_tab)

        # 分词
        naming_line_token = word_tokenize(tmp)
        api_token = word_tokenize(cs)

        logger.debug(f"推荐点所在行分词：{json.dumps(naming_line_token)}##api分词: {api_token}##计算这些分词的共现次数")

        total = 0.0
        conum = 0.0
        score = 0.0
        for w in api_token:
            # api的出现次数
            total = total + self.get_total(w, naming_context, files)
            for n in naming_line_token:
                # api和当前行分词的出现次数
                conum += self.get_conum(w, n, naming_context, files)
        if total != 0:
            total = float(total)
            conum = float(conum)
            score = float(conum / total)  # 计算共现分数
        return score

    # 计算w在所有文件中的出现次数，files为不导入当前文件的文件列表？
    def get_total(self, w, naming_context, files):
        ret = 0.0
        for fi in files:
            # 在其他文件中，w的出现次数
            key = f'{w}##{fi}'
            if key in self.proj_token_count:
                ret += self.proj_token_count[key]
        ret += naming_context.count(w)  # 在当前文件中w的出现次数
        return ret

    # 计算在其他文件中w和n在同个文件同一行的共现数量
    def get_conum(self, w, n, naming_context, files):
        ret = 0.0
        for fi in files:
            k1 = f'{w}##{fi}'
            k2 = f'{n}##{fi}'
            if k1 in self.proj_token_no and k2 in self.proj_token_no:
                x1 = self.proj_token_no[k1]
                y1 = self.proj_token_no[k2]
                # w和n在相同行出现的次数？
                ctis = [x for x in x1 if x in y1]
                ret += float(len(ctis))
        return ret

    # 获取候选api和推荐点前上下文的在其他文件中的共现分数
    def get_conum_scores(self, aps, naming_context, file_path: CleanFilePath):
        conum_scores = {}

        print(file_path)
        fi = re.sub(r'\.py', '', str(file_path))
        index = fi.rfind('/')
        cur_name = fi[index + 1:]

        files = []
        for k, v in self.proj_depends.items():
            k = CleanFilePath(k)
            if k == file_path:
                continue
            flag = 0
            for imports in v:
                if cur_name in imports:
                    flag = 1
                    break
            if flag == 0:
                files.append(k)
        for api in aps:
            if api.startswith('__') or re.match('[A-Z0-9_]+$', api) or api.strip() == '_':
                continue
            con_ret = self.get_conum_of_con(api, naming_context, files)
            conum_scores[api] = con_ret
        return conum_scores

    def get_conum_of_con(self, api, naming_context, files: List[CleanFilePath]):
        del_estr = string.punctuation + string.digits
        replace = " " * len(del_estr)
        tran_tab = str.maketrans(del_estr, replace)

        code = naming_context.strip()
        lines = code.split('\n')

        rets = 0.0
        # 处理api和每一行各个token的共现分数
        for i in range(0, len(lines)):
            tmp = lines[i].translate(tran_tab)
            cs = api.translate(tran_tab)

            line_token = word_tokenize(tmp)
            api_token = word_tokenize(cs)

            total = 0.0
            for w in api_token:
                total = total + self.get_total_infile(w, files)
            conum = 0.0
            # 计算每个w和每个item的共现次数
            for w in api_token:
                for item in line_token:
                    conum = conum + self.get_conum_infile(w, item, files)
            if total != 0:
                total = float(total)
                conum = float(conum)
                score = float(conum / total)
                rets += float(i + 1) * score  # 距离越远，分数越高？
        context_ret = float(float(rets) / float(len(lines) + 1.0))
        return context_ret

    # w出现的文件数量？
    def get_total_infile(self, w, files):
        ret = 0.0
        for fi in files:
            key = f'{w}##{fi}'
            if key in self.proj_token_count:
                ret += 1.0
        return ret

    # w和item共同出现的文件数量？
    def get_conum_infile(self, w, item, files):
        ret = 0.0
        for fi in files:
            k1 = f'{w}##{fi}'
            k2 = f'{item}##{fi}'
            if k1 in self.proj_token_no and k2 in self.proj_token_no:
                ret += 1.0
        return ret


if __name__ == '__main__':
    feature_encoder = FeatureEncoder()
    feature_encoder.encode_features()
    # input() # 子进程的错误不用管，不影响主进程执行，因为input()运行完后子进程才报错的
