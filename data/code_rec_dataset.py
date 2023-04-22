# -*- coding=utf-8 -*-
# @Time: 2022/12/12 16:45
# @Author: Jeremy
# @File: code_rec_dataset.py
# @Software: PyCharm
from concurrent.futures import ProcessPoolExecutor
import json
import torch
import re
import ast
import string
from typing import List, Literal

from tools import get_dataflow
from data_utils import DataUtils
from common_utils import CommonUtils
from feature_encoder import FeatureEncoder
from torch.utils.data.dataset import Dataset, T_co
from tools.path_utils import *
from tools.data.data_collator import get_concat_batch_inputs
from config import config


# --------------------------------------------------
# LOG CONFIG
# --------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)

console = logging.StreamHandler()
console.setLevel(level=logging.INFO)  # 控制台输出INFO级别以上的信息
logger.addHandler(console)

file = logging.FileHandler(config["log"]["info"], encoding="utf-8", mode="w")
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


class CodeRecDataset(Dataset):
    def __init__(self, proj_name=config["project"]["proj_name"], split="train"):
        # , datasets_path="datasets"
        # , raw_data_dir="raw-data", clean_data_dir="clean-data",
        #  json_data_dir="json-data", encoded_data_dir="encoded-data"
        # self.raw_data_path = Path(datasets_path, raw_data_dir)
        # self.clean_data_path = Path(datasets_path, clean_data_dir)
        # self.json_data_path = Path(datasets_path, json_data_dir)
        # self.encoded_data_path = Path(datasets_path, encoded_data_dir)
        self.test_rec_info = None
        self.split = split
        self.size = None
        self.labels = None
        self.codes = None
        self.asts = None
        self.nls = None
        self.dataflow_scores = None
        self.tosim_scores = None
        self.line_scores = None
        self.conum_scores = None
        self.raw_data_path = RawFilePath()
        # self.clean_data_path = CleanFilePath()
        # self.json_data_path = JsonFilePath()
        # self.encoded_data_path = EncodedFilePath()

        self.proj_name = proj_name

        self.proj_file_list = list(DataUtils.get_file_list(RawFilePath(self.proj_name)))
        self.clean_file_list = self.clean_proj_files()
        self.clean_file_list.sort()
        self.train_list, self.test_list = self.train_test_split()

        self.all_apis = {}
        self.all_apis_add = []

        self.none_type_count = 0
        self.any_type_count = 0
        self.ok_type_count = 0
        # self.type_info_table = PrettyTable()
        # self.type_info_table.field_names = ["NoneTypeCount", "AnyTypeCount", "OKTypeCount"]
        if split == "train":
            self.encode_train_features()
        elif split in ["test", "valid"]:
            self.encode_test_features()

    def __getitem__(self, index) -> T_co:
        return self.codes[index], self.asts[index], self.nls[index], self.dataflow_scores[index],\
               self.line_scores[index], self.tosim_scores[index], self.conum_scores[index], self.labels[index]

    def __len__(self):
        return self.size

    def clean_proj_files(self) -> List[CleanFilePath]:

        ret_list = []

        for file in self.proj_file_list:
            source_code = file.read_text('utf-8')
            source_code_without_comment = DataUtils.remove_comments_and_docstrings(source_code)

            # clean_file_path = self.clean_data_path.joinpath(file.relative_to(self.raw_data_path))
            clean_file_path = file.change_to_path(CleanFilePath)
            clean_file_path.parent.mkdir(parents=True, exist_ok=True)

            ret_list.append(clean_file_path)
            clean_file_path.write_text(source_code_without_comment, 'utf-8')

            file.get_logger().info('-' * 100)
            file.get_logger().info(f"{file}文件清除注释完毕，存入{clean_file_path}中")

        return ret_list

    def train_test_split(self):
        train_len = int(len(self.clean_file_list) / 10 * 8)
        train_list = self.clean_file_list[:train_len]
        test_list = self.clean_file_list[train_len:]
        logger.debug(f"训练集：{train_list}")
        logger.debug(f"测试集：{test_list}")
        return train_list, test_list

    def encode_train_features(self):
        self.collect_train_features()
        feature_encoder = FeatureEncoder(split="train")
        train_pyart_input, train_spt_code_input, train_labels = feature_encoder.encode_features()
        self.codes = train_spt_code_input["all_codes"]
        self.asts = train_spt_code_input["all_asts"]
        self.nls = train_spt_code_input["all_nls"]
        self.dataflow_scores = train_pyart_input["all_dataflow_scores"]
        self.tosim_scores = train_pyart_input["all_tosim_scores"]
        self.line_scores = train_pyart_input["all_line_scores"]
        self.conum_scores = train_pyart_input["all_conum_scores"]
        self.labels = train_labels

        assert len(self.codes) == len(self.asts) == len(self.nls) == len(self.dataflow_scores) == len(self.line_scores)\
               == len(self.tosim_scores) == len(self.conum_scores) == len(self.labels)
        self.size = len(self.labels)

    def encode_test_features(self):
        self.collect_test_features()
        feature_encoder = FeatureEncoder(split="test")
        test_pyart_input, test_spt_code_input, test_labels, test_rec_info = feature_encoder.encode_features()
        self.codes = test_spt_code_input["all_codes"]
        self.asts = test_spt_code_input["all_asts"]
        self.nls = test_spt_code_input["all_nls"]
        self.dataflow_scores = test_pyart_input["all_dataflow_scores"]
        self.tosim_scores = test_pyart_input["all_tosim_scores"]
        self.line_scores = test_pyart_input["all_line_scores"]
        self.conum_scores = test_pyart_input["all_conum_scores"]
        self.labels = test_labels
        self.test_rec_info = test_rec_info

        assert len(self.codes) == len(self.asts) == len(self.nls) == len(self.dataflow_scores) == len(self.line_scores)\
               == len(self.tosim_scores) == len(self.conum_scores) == len(self.labels)
        self.size = len(self.labels)

    def collect_features(self, mode: Literal["train", "test"], data_list: List[CleanFilePath]):
        if processes_num > 0:
            with ProcessPoolExecutor(processes_num) as executor, \
                    open(config["project"][f"proj_{mode}_json_list"].format(self.proj_name), "w") as f:
                for features in executor.map(self.deal_with_file, data_list):
                    f.write(features)
        else:
            with open(config["project"][f"proj_{mode}_json_list"].format(self.proj_name), "w") as f:
                for features in map(self.deal_with_file, data_list):
                    f.write(features)
        with open(config["project"][f"proj_{mode}_json_list_files"].format(self.proj_name), 'w') as f:
            for file_path in data_list:
                print(file_path.change_to_path(JsonListEncodedFilePath), file=f)

    def collect_train_features(self):
        return self.collect_features("train", self.train_list)

    def collect_test_features(self):
        return self.collect_features("test", self.test_list)

    def deal_with_file(self, file_path: CleanFilePath):
        file_path.get_logger().debug("-" * 100)
        file_path.get_logger().debug(f"当前处理文件{file_path}")

        result_file_path = file_path.change_to_path(JsonListEncodedFilePath)
        if result_file_path.is_file():
            return result_file_path.read_text()
        result_file_path.parent.mkdir(parents=True, exist_ok=True)
        result_file_path.touch(exist_ok=True)

        current_module_apis, current_def_apis = self.get_module_methods(
            file_path.change_to_path(RawFilePath))

        self.all_apis = self.get_all_apis(current_module_apis)

        current_all_apis = self.all_apis['all_apis']
        current_all_apis.extend(self.all_apis_add)
        current_all_apis = list(set(current_all_apis))
        self.all_apis['all_apis'] = current_all_apis

        file_features = self.get_file_features(file_path, current_def_apis)
        result_file_path.write_text(file_features)

        return file_features

    def get_module_methods(self, file_path: RawFilePath):

        logger = file_path.get_logger()
        logger.debug("-" * 100)
        logger.debug(f"获取{file_path}文件中导入的包中以及自身定义的所有API")

        all_module_apis = {}
        current_def_apis = {}

        current_def_class = []
        current_def_funcs = []
        current_modules = []
        current_items = {}
        current_alias = {}

        json_data_file_path = file_path.change_to_path(JsonFilePath)
        json_data_file_path.make_parent_dirs()
        tree = ast.parse(file_path.read_text())

        for node in ast.walk(tree):

            if isinstance(node, ast.ClassDef):

                module_name = DataUtils.get_file_module_name(file_path, self.raw_data_path)
                item_name = node.name
                current_def_class.append(item_name)

                logger.debug(f"收集当前{module_name}中的{item_name}的所有API")

                current_class_apis = DataUtils.get_item_funcs(module_name, item_name)
                if current_class_apis == {}:
                    logger.debug(f"当前{module_name}中的{item_name}的所有API收集失败")
                else:
                    logger.debug(f"当前{module_name}中的{item_name}的所有API收集成功")
                    current_def_apis.update(current_class_apis)

            elif isinstance(node, ast.Import):

                for node_name in node.names:

                    module_name = node_name.name
                    current_modules.append(module_name)

                    if node_name.asname is None:

                        logger.debug(f"收集{module_name}中的所有API")

                        module_apis = DataUtils.get_module_funcs(module_name)
                        if module_apis == {}:
                            logger.debug(f"{module_name}中的所有API收集失败")
                        else:
                            logger.debug(f"{module_name}中的所有API收集成功")
                            all_module_apis.update(module_apis)
                    else:
                        alias_name = node_name.asname
                        current_alias.update({module_name: alias_name})

                        logger.debug(f"收集{module_name}##{alias_name}别名的所有API")

                        module_apis = DataUtils.get_alias_funcs(module_name, alias_name)
                        if module_apis == {}:
                            logger.debug(f"{module_name}##{alias_name}别名的所有API收集失败")
                        else:
                            logger.debug(f"{module_name}##{alias_name}别名的所有API收集成功")
                            all_module_apis.update(module_apis)

            elif isinstance(node, ast.ImportFrom):

                if node.level == 0:
                    module_name = node.module
                    current_modules.append(module_name)
                else:
                    module_name = DataUtils.get_local_module(file_path, self.raw_data_path, node.module, node.level)
                    current_modules.append(module_name)

                for node_name in node.names:
                    if node_name.asname is None:
                        item_name = node_name.name
                        current_items.setdefault(module_name, [])
                        current_items[module_name].append(item_name)

                        logger.debug(f"收集{module_name}中的{item_name}的所有API")

                        module_apis = DataUtils.get_item_funcs(module_name, item_name)
                        if module_apis == {}:
                            logger.debug(f"{module_name}中的{item_name}的所有API收集失败")
                        else:
                            logger.debug(f"{module_name}中的{item_name}的所有API收集成功")
                            all_module_apis.update(module_apis)
                    else:
                        item_name = node_name.name
                        alias_name = node_name.asname
                        current_items.setdefault(module_name, [])
                        current_items[module_name].append(f"{item_name}##{alias_name}")

                        logger.debug(f"收集{module_name}中的{item_name}##{alias_name}的所有API")

                        module_apis = DataUtils.get_item_alias_funcs(module_name, item_name, alias_name)
                        if module_apis == {}:
                            logger.debug(f"{module_name}中的{item_name}##{alias_name}的所有API收集失败")
                        else:
                            logger.debug(f"{module_name}中的{item_name}##{alias_name}的所有API收集成功")
                            all_module_apis.update(module_apis)

            else:
                continue

            logger.debug("-" * 100)

        with open(json_data_file_path.change_to_path(ModuleApisJsonFilePath), "w") as f:
            json.dump(all_module_apis, f, indent=4, ensure_ascii=True)
        with open(json_data_file_path.change_to_path(DefApisJsonFilePath), "w") as f:
            json.dump(current_def_apis, f, indent=4, ensure_ascii=True)

        print(current_modules)
        print(current_alias)
        print(current_items)
        print(current_def_class)

        return all_module_apis, current_def_apis

    def get_file_features(self, file_path: CleanFilePath, current_def_apis):

        logger = file_path.get_logger()
        logger.info('-' * 100)
        logger.info(f"提取{file_path}文件的推荐点前的代码上下文，以及候选API")

        pre_code = ''
        line_no = 0
        rec_points = []

        try_block_count = 0
        try_block_indent_list = []

        with open(file_path, "r", encoding='utf-8') as f:
            code_lines = f.readlines()

        feature_str = ""

        for code_line in code_lines:
            line_no += 1

            try_block_count, try_block_indent_list = self.get_try_info(
                code_line, try_block_count, try_block_indent_list)

            rec_point = re.findall(r'[a-zA-Z0-9_\.\[\]]+\.[a-zA-Z0-9\_]+\(.*\)', code_line)
            if len(rec_point) == 0:
                pre_code += code_line
                continue

            # TODO: 可能一行有多个推荐点
            rec_point = rec_point[0]
            caller, callee, callee_with_bracket = DataUtils.caller_callee_split(rec_point)

            logger.debug(f"疑似推荐点: {file_path}##{line_no}##{rec_point}")
            logger.debug(f"caller: {caller}##callee: {callee}##callee_with_bracket: {callee_with_bracket}")

            if callee.startswith('_') or re.match('[A-Z0-9_]+$', callee) or callee.strip() == '_' \
                    or re.match('[A-Z]+[A-Za-z]+', callee):
                pre_code += code_line
                logger.debug(f"callee有问题，跳过推荐点：{file_path}##{line_no}##{rec_point}")
                continue

            rec_point_reconstruction = caller + "." + callee
            # 如果是已统计的推荐点，将当前推荐点加入上下文
            # TODO: 这样会不会丢失后面推荐点的上下文信息？
            if rec_point_reconstruction in rec_points:
                pre_code += code_line
                logger.debug(f"当前推荐点已统计过，跳过推荐点：{file_path}##{line_no}##{rec_point}")
                continue
            else:
                rec_points.append(rec_point_reconstruction)

            context_for_ast = pre_code + code_line.replace(callee_with_bracket, "[API_POINT]")
            context_for_data_flow, context_for_type_inference = self.get_context_info(
                pre_code, code_line, caller, callee_with_bracket, try_block_indent_list)

            caller_type = DataUtils.get_caller_type(context_for_type_inference)

            logger.debug(f"推断出的类型：{caller_type}")

            # 各种类型的数量
            if caller_type == 'None':
                self.none_type_count += 1
            elif caller_type == 'Any' or caller_type == 'nothing':
                self.any_type_count += 1
            else:
                self.ok_type_count += 1
            logger.debug(f"类型推断情况##None_type_count: {self.none_type_count}##Any_type_count: {self.any_type_count}"
                         f"##Ok_type_count: {self.ok_type_count}")

            candidate_apis = []
            caller, caller_type = self.reconstruct_caller_type(caller, caller_type)

            collected_apis = self.get_other_candidates(caller_type, caller, file_path, current_def_apis)
            for k, v in collected_apis.items():
                candidate_apis.extend(v)

            if len(candidate_apis) == 0:
                logger.debug(f"当前推荐点候选API为0，跳过推荐点：{file_path}##{line_no}##{rec_point}")
                pre_code += code_line
                continue

            if callee in candidate_apis:
                # IV是登录词，即字典中存在的词（候选api列表中包含正确的callee）
                logger.debug(f"当前推荐点候选API包含正确的API")
                logger.debug('API IV')
            else:
                # OOV是未登录词，说明收集到的候选api中没有正确的api
                logger.debug(f"当前推荐点候选API未包含正确的API，跳过推荐点：{file_path}##{line_no}##{rec_point}，并将当前推荐点加入file_apis")
                logger.debug('API OOV')
                self.all_apis_add.append(callee)
                # all_apis是字典，取出一个列表
                tmpx = self.all_apis['all_apis']
                tmpx.extend(self.all_apis_add)  # 将候选api中没有的正确api放入总的api中
                tmpx = list(set(tmpx))
                self.all_apis['all_apis'] = tmpx
                pre_code += code_line
                continue

            # print('[Process[2] : Constructing dataflow hints.]')
            current_dataflow = get_dataflow.get_current_dataflow2(context_for_data_flow, caller)
            # 数据流构建不成功也跳过当前推荐点
            if len(current_dataflow) == 0:
                logger.debug(f"当前推荐点数据流构造失败，跳过推荐点：{file_path}##{line_no}##{rec_point}")
                pre_code += code_line
                continue
            max_flow = max(current_dataflow, key=len)
            print(max_flow)

            # 数据流分数和相似度分数
            # dataflow_scores = get_dataflow.get_dataflow_scores(candidate_apis, max_flow, current_dataflow,
            #                                                    caller_type, callee)
            # tosim_scores = get_dataflow.get_tosim_scores(candidate_apis, max_flow, current_dataflow, caller_type,
            #                                              callee)

            try:
                # 去掉代码行中正确的api
                naming_line = re.sub(callee, '', code_line)
            except Exception as err:
                logger.debug(f"当前推荐点所在行替换正确API失败，跳过推荐点：{file_path}##{line_no}##{rec_point}##{err}")
                pre_code += code_line
                continue
            naming_context = pre_code

            punc = re.sub(r"[\"'_]", "", string.punctuation)
            code_tokens = [i.strip("\n\t ") for i in re.split(rf"(\b|[{punc}]|(['\"])\w+\2)", context_for_ast) if i and
                           i.strip("\n\t ")]
            feature_info = {
                "rec_point": rec_point,
                "positive_api": callee,
                "caller_type": caller_type,
                "candidate_apis": candidate_apis,
                "code_tokens": code_tokens,
                "context_for_ast": context_for_ast,
                "context_for_data_flow": context_for_data_flow,
                "context_for_type_inference": context_for_type_inference,
                "naming_line": naming_line,
                "naming_context": naming_context,
                "file": str(file_path.get_relative_path()),
                "current_data_flow": current_dataflow,
                "max_flow": max_flow
            }
            feature_str += json.dumps(feature_info) + "\n"
        return feature_str

    @staticmethod
    def get_try_info(code_line, try_block_count, try_block_indent_list):
        if "try:" in code_line:
            try_block_count += 1
            try_block_indent_list.append(CommonUtils.get_blank(code_line))
        elif try_block_count > 0 and ('except' in code_line or 'finally:' in code_line):
            blank, blank_length = CommonUtils.get_blank(code_line)
            # 去掉匹配的try-catch块，剩下的不匹配的在后面补充匹配
            for i in range(len(try_block_indent_list) - 1, -1, -1):
                if try_block_indent_list[i][1] == blank_length:
                    try_block_count -= 1
                    del try_block_indent_list[i]
        return try_block_count, try_block_indent_list

    @staticmethod
    def get_context_info(pre_code, code_line, caller, callee_with_bracket, try_block_indent_list):
        last_line = code_line.replace(callee_with_bracket, "unknown_api()")
        original_context = pre_code.strip()

        if original_context.endswith(','):
            new_context = original_context[:-1]

            final_context = DataUtils.complete_context_bracket(new_context)
            context_for_data_flow = final_context + '\n' + last_line

            # 包含api推荐点的前面一行
            line_before_rec_point = original_context.split('\n')[-1]
            for i in range(0, len(line_before_rec_point)):
                if line_before_rec_point[i] != ' ':  # 找到第一个非空格的地方
                    break
            # line[:i - 4]是空格？最后一行有逗号，所以下一行缩进减少？
            context_for_type_inference = final_context + '\n' + code_line[
                                                                : i - 4] + 'reveal_type(' + caller + ')'
        elif original_context.endswith('(') or original_context.endswith('{') or original_context.endswith('['):
            new_context = original_context

            final_context = DataUtils.complete_context_bracket(new_context)  # 补全括号
            context_for_data_flow = final_context + '\n' + last_line

            line_before_rec_point = original_context.split('\n')[-1]
            for i in range(0, len(line_before_rec_point)):
                if line_before_rec_point[i] != ' ':
                    break
            # 跟前面一行保持相同缩进
            context_for_type_inference = final_context + '\n' + code_line[:i] + 'reveal_type(' + caller + ')'
        else:
            new_context = original_context

            final_context = DataUtils.complete_context_bracket(new_context)
            context_for_data_flow = final_context + '\n' + last_line

            for i in range(0, len(code_line)):
                if code_line[i] != ' ':
                    break
            context_for_type_inference = final_context + '\n' + code_line[:i] + 'reveal_type(' + caller + ')'

        # 如果推荐点包含在try块内，则在上面构造的用于类型推断的上下文的基础上补充try-exception块
        if len(try_block_indent_list) > 0:
            context_for_type_inference = DataUtils.complete_context_try(context_for_type_inference,
                                                                        try_block_indent_list)
        return context_for_data_flow, context_for_type_inference

    @staticmethod
    def reconstruct_caller_type(caller, caller_type):
        # 如果类型推断失败，构造一个类似类型用来获取候选API
        if caller_type == "None" or caller_type == "Any":
            if caller == "self":
                caller_type = "self_def_class"
            elif caller == "str" or caller == "s" or caller == "string":
                caller_type = "str"
            elif caller == "log":
                caller_type = "logging.Logger"
        elif "import" in caller_type:
            caller_type = caller_type.split(" ")[-1]
        return caller, caller_type

    # 根据重构得到的类型来获取候选API
    def get_other_candidates(self, caller_type, caller, file: CleanFilePath, current_def_apis):
        logger = file.get_logger()
        logger.debug("-" * 100)
        logger.debug(f"caller_type: {caller_type}##caller: {caller}##file: {file}")

        if caller_type.startswith('Type['):
            caller_type = caller_type[5:-1]
        candidates = {}

        # 收集推断出类型的候选api列表
        if caller_type == 'self_def_class':
            logger.debug(f"当前是{caller_type}类型，从文件本身定义的类中获取候选API")
            candidates = current_def_apis
        elif caller_type == 'str':
            candidates = {caller: dir(str)}
        elif re.match(r'List\[.*\]', caller_type):
            candidates = {caller: dir(list)}
        elif re.match(r'Dict\[.*\]', caller_type) or caller_type == "dict":
            candidates = {caller: dir(dict)}
            logger.debug(f"当前是Dict类型，收集到的API为{json.dumps(candidates)}")
        elif re.match(r'Set\[.*\]', caller_type) or caller_type == 'set':
            candidates = {caller: dir(set)}
        elif caller_type == 'bool':
            candidates = {caller: dir(bool)}
        # 收集Union中每个对象的api列表
        elif re.match(r'Union\[.*\]', caller_type):
            caller_type = caller_type + 'end'
            contents = CommonUtils.get_middle_str(caller_type, 'Union[', ']end')
            contents = re.sub(r'\[.*\]', '', contents)
            union_type_list = contents.split(',')
            tmp_apis = []
            # 将Union里的全部类型的候选api列表收集
            for union_type in union_type_list:
                union_type = union_type.strip()
                if union_type == 'Any' or union_type == 'nothing':
                    continue
                type_apis_dict = self.get_other_candidates(union_type, caller, file, current_def_apis)
                for k, v in type_apis_dict.items():
                    tmp_apis.extend(v)  # 在列表后面追加列表
            candidates = {caller: tmp_apis}
        elif re.match(r'Optional\[.*\]', caller_type):
            candidates = {}
        # 处理包含子模块的调用
        elif '.' in caller_type:
            caller_type = re.sub(r'\[.*\]', '', caller_type)
            module_index = caller_type.rfind('.')
            module_name = caller_type[:module_index]  # 除了最后一个.后面的内容的前面的内容
            module_item = caller_type[module_index + 1:]  # item是最后的一个.后面的内容
            candidates = DataUtils.get_item_funcs(module_name, module_item)
        elif caller_type == 'Any' or caller_type == 'None' or caller_type == 'nothing':
            candidates = self.all_apis
            logger.debug(f"当前是{caller_type}类型，从all_apis中获取得到的API数量为{len(candidates['all_apis'])}")
            return candidates
        elif re.match('[a-zA-Z0-9_]+', caller_type):
            if caller_type in current_def_apis.keys():
                candidates = {caller_type: current_def_apis[caller_type]}
            else:
                module_name = DataUtils.get_file_module_name(file, config["datasets"]["clean-data"])
                module_item = caller_type
                candidates = DataUtils.get_item_funcs(module_name, module_item)
            logger.debug(f"当前是{caller_type}类型，从文件本身定义的类中或者外部库中获取候选API")

        # 没有从已知类型中收集到api，那么就从typeshed api里收集候选api
        if len(candidates) == 0:
            typeshed_apis = DataUtils.get_typeshed_apis(caller_type)
            candidates.update({caller: typeshed_apis})
        for k, v in candidates.items():
            dag = []
            for j in range(0, len(v)):
                # 去掉以__开头的变量
                if not v[j].startswith('__'):
                    dag.append(v[j])
            candidates[k] = dag
        return candidates

    def get_all_apis(self, current_module_apis):
        # TODO:count all apis,including module_apis,builtin_apis,proj_apis
        ret_apis = []

        for k, v in current_module_apis.items():
            for f in v:
                if not f.startswith('__') and not re.match('[A-Z0-9]+', f) and f not in ret_apis:
                    ret_apis.append(f)

        with open(config["project"]["proj_json"].format(self.proj_name)) as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip()
            index = line.rfind('.')
            item = line[index + 1:]
            if not item.startswith('__') and item not in ret_apis:
                ret_apis.append(item)

        with open(config["other"]["builtin_path"]) as f:
            lines = f.readlines()
        for line in lines:
            item = line.strip()
            if not item not in ret_apis:
                ret_apis.append(item)

        return {'all_apis': ret_apis}


def collate_fn(batch, code_vocab, nl_vocab, ast_vocab):
    model_inputs = {}
    code_raw, ast_raw, nl_raw, dataflow_score, line_score, tosim_score, conum_score, label = map(list, zip(*batch))

    model_inputs['input_ids'], model_inputs['attention_mask'] = get_concat_batch_inputs(
        code_raw=code_raw,
        code_vocab=code_vocab,
        max_code_len=256,
        ast_raw=ast_raw,
        ast_vocab=ast_vocab,
        max_ast_len=32,
        nl_raw=nl_raw,
        nl_vocab=nl_vocab,
        max_nl_len=64
    )
    model_inputs['dataflow_score'] = torch.tensor(dataflow_score, dtype=torch.float32)
    model_inputs['line_score'] = torch.tensor(line_score, dtype=torch.float32)
    model_inputs['tosim_score'] = torch.tensor(tosim_score, dtype=torch.float32)
    model_inputs['conum_score'] = torch.tensor(conum_score, dtype=torch.float32)
    model_inputs['labels'] = torch.tensor(label, dtype=torch.float32)

    return model_inputs


if __name__ == '__main__':
    code_rec_dataset = CodeRecDataset()
    code_rec_dataset.collect_test_features()
    test_data = code_rec_dataset[1]
    print(test_data)
    # code_rec_dataset.deal_with_file(
    #     r"/root/autodl-tmp/jeremy/Programs/Python/python-projects/PyBartRec/data/clean-data/flask/app.py")
    # code_rec_dataset.get_module_methods(r"D:\Jeremy\OneDrive\工作\pycharm-workspace\PyBartRec\data\clean-data\flask\app"
    #                                     r".py")
