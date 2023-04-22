# -*- coding=utf-8 -*-
# @Time: 2022/12/12 21:53
# @Author: Jeremy
# @File: data_utils.py
# @Software: PyCharm
import json
from io import StringIO
from pathlib import Path
import shlex
import subprocess
import tokenize
import re
import os
import sys
import logging
import importlib
import tempfile
from typing import Union, Generator, TypeVar
from common_utils import CommonUtils
from tree_sitter import Language, Parser
from config import config

runtime_pip = config["other"]["runtime_pip"]

# --------------------------------------------------
# LOG CONFIG
# --------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)

console = logging.StreamHandler()
console.setLevel(level=logging.INFO)  # 控制台输出INFO级别以上的信息
logger.addHandler(console)

file = logging.FileHandler(config["log"]["data_utils"], encoding="utf-8", mode="w")
file.setLevel(level=logging.DEBUG)  # 文件输出DEBUG级别以上信息（全部信息）
formatter = logging.Formatter('[%(asctime)s | %(filename)s | line %(lineno)d] - %(levelname)s: %(message)s')
file.setFormatter(formatter)
logger.addHandler(file)

logger.debug('-' * 100)
logger.debug('Start print log')

PathType = TypeVar('PathType', bound=Path)
class DataUtils:
    def __init__(self):
        pass

    @classmethod
    def get_file_list(cls, base_path: PathType) -> Generator[PathType, None, None]:
        if isinstance(base_path, str):
            base_path = Path(base_path)
        # glob.rglob返回匹配模板的路径的迭代器
        for file in base_path.rglob("*.py"):
            if file.is_file():
                yield file

    @classmethod
    def remove_comments_and_docstrings(cls, source):
        """
        Remove docs and comments from source string.
        Thanks to authors of GraphCodeBERT
        from: https://github.com/microsoft/CodeBERT/blob/master/GraphCodeBERT/codesearch/parser/utils.py#L4

        Args:
            source (str): Source code string
            lang (str): Source code language

        Returns:
            str: Source string

        """
        try:
            io_obj = StringIO(source)
            out = ""
            prev_token_type = tokenize.INDENT
            last_lineno = -1
            last_col = 0
            for tok in tokenize.generate_tokens(io_obj.readline):
                token_type = tok[0]
                token_string = tok[1]
                start_line, start_col = tok[2]
                end_line, end_col = tok[3]
                # l_text = tok[4]
                if start_line > last_lineno:
                    last_col = 0
                if start_col > last_col:
                    out += (" " * (start_col - last_col))
                # Remove comments:
                if token_type == tokenize.COMMENT:
                    pass
                # This series of conditionals removes docstrings:
                elif token_type == tokenize.STRING:
                    if prev_token_type != tokenize.INDENT:
                        # This is likely a docstring; double-check we're not inside an operator:
                        if prev_token_type != tokenize.NEWLINE:
                            if start_col > 0:
                                out += token_string
                else:
                    out += token_string
                prev_token_type = token_type
                last_col = end_col
                last_lineno = end_line
            temp = []
            for x in out.split('\n'):
                if x.strip() != "":
                    temp.append(x)
            return '\n'.join(temp)
        except Exception:
            return source

    # 将推荐点分割为caller和callee
    @classmethod
    def caller_callee_split(cls, rec_point):
        # 推荐点形式是builder.close()，要去掉close后面的括号
        rec_without_bracket = re.sub('\(.*\)', '', rec_point)
        pindex = rec_without_bracket.rfind('.')
        return rec_without_bracket[:pindex], rec_without_bracket[pindex + 1:], rec_point[pindex + 1:]

    # 将推荐点前上下文字符串都去除，并且补全括号
    @classmethod
    def complete_context_bracket(cls, context):
        context_lines = context.split('\n')

        i = 0
        # 找到最后一个方法定义那行
        for i in range(len(context_lines) - 1, -1, -1):
            if context_lines[i].strip().startswith('def'):
                break

        last_function_context = ''
        for j in range(i, len(context_lines)):
            last_function_context += context_lines[j] + '\n'

        # 将方法块中的字符串都去掉？
        function_context_wo_string = re.sub('\'[\\\[\]\(\)\{\}A-Za-z0-9_\,\:]+\'', '', last_function_context)
        function_context_wo_string = re.sub('\"[\\\[\]\(\)\{\}A-Za-z0-9_\,\:]+\"', '', function_context_wo_string)

        lk = function_context_wo_string.count('(')
        rk = function_context_wo_string.count(')')
        ll = function_context_wo_string.count('[')
        rl = function_context_wo_string.count(']')
        ld = function_context_wo_string.count('{')
        rd = function_context_wo_string.count('}')
        kc = lk - rk
        lc = ll - rl
        dc = ld - rd

        add_bracket = ''
        if kc == lc == dc == 0:  # 括号数量匹配，返回原上下文
            return context
        else:
            # ks存储括号
            ks = ''
            for i in range(0, len(function_context_wo_string)):
                c = function_context_wo_string[i]
                # 找到有括号的字符
                if re.match('[\(\)\[\]\{\}]', c):
                    ks += c
            # 去掉匹配的括号，因为括号中可能包含括号，说以每一轮去除括号后，都会剩下最外层的括号，一直去除到没有匹配的括号后为止
            while '{}' in ks or '[]' in ks or '()' in ks:
                while '()' in ks:
                    ks = re.sub('\[\]', '', ks)
                    ks = re.sub('\{\}', '', ks)
                    ks = re.sub('\(\)', '', ks)
                while '[]' in ks:
                    ks = re.sub('\{\}', '', ks)
                    ks = re.sub('\(\)', '', ks)
                    ks = re.sub('\[\]', '', ks)
                while '{}' in ks:
                    ks = re.sub('\[\]', '', ks)
                    ks = re.sub('\(\)', '', ks)
                    ks = re.sub('\{\}', '', ks)
            # 从后往前补充缺失的括号
            for i in range(len(ks) - 1, -1, -1):
                if ks[i] == '(':
                    add_bracket += ')'
                elif ks[i] == '[':
                    add_bracket += ']'
                else:
                    add_bracket += '}'
            return context + add_bracket

    @classmethod
    def complete_context_try(cls, context, try_block_indent_list):
        ret_context = context
        for i in range(len(try_block_indent_list) - 1, -1, -1):
            ret_context += '\n' + try_block_indent_list[i][0] + 'except Exception:\n' + try_block_indent_list[i][
                0] + '	' + 'pass'
        return ret_context

    # 利用pytype获取类型，context是修改后的推荐点前的上下文（包含要推断的caller类型）
    @classmethod
    def get_caller_type(cls, context):
        # file_for_pytype = file_path.with_suffix(".tmp.py")
        with tempfile.NamedTemporaryFile("w", encoding='utf-8', suffix=".py",
                                         delete=False) as f:
            file_for_pytype = f.name
            f.write(context)
        # file_for_pytype.write_text(context)
        # with open(file_for_pytype, 'w+') as f:
        #     f.write(context)

        try:
            # 利用pytype推荐类型
            pytype_output = subprocess.run(shlex.split(f'pytype {file_for_pytype}'), encoding='utf-8',
                                           capture_output=True)
            # os.system(f'pytype {file_for_pytype} > pytype_log.txt')
        except Exception as err:
            logging.exception("pytype运行出错", exc_info=err)
        os.remove(file_for_pytype)
        lines = pytype_output.stdout.splitlines()
        # with open('pytype_log.txt') as f:
        #     lines = f.readlines()

        inferred_type = 'None'
        for line in lines:
            if '[reveal-type]' in line:
                tp = line.split(':')[1]
                inferred_type = re.sub(r'\[reveal\-type\]', '', tp)
                break
        return inferred_type.strip()

    @classmethod
    def get_module_funcs(cls, module_name):

        logger.debug(f"get_module_funcs##module_name: {module_name}")

        module_name = module_name.strip()

        # 如果当前的api列表没有这个模块的api则要引入这个模块，引入这个模块才能用dir函数收集这个模块的方法列表
        try:
            imported_module = importlib.import_module(module_name)
        # 引入失败则安装这个模块
        except Exception as err:
            # 有"."号如models.bert说明需要安装根模块models，只需要安装根模块即可
            logger.debug(f"第一次引入模块失败，尝试安装{module_name}模块##{err}")
            try:
                if '.' in module_name:
                    index = module_name.find('.')
                    root_module = module_name[:index]
                    os.system(f'{runtime_pip} install {root_module}')
                else:
                    os.system(f'{runtime_pip} install {module_name}')
                imported_module = importlib.import_module(module_name)
            except Exception as err:
                logger.debug(f"安装{module_name}模块失败##{err}##返回空列表")
                return {}

        module_apis = dir(imported_module)  # dir函数返回这个模块的方法，属性列表

        return {module_name: module_apis}

    @classmethod
    def get_alias_funcs(cls, module_name, alias_name):

        logger.debug(f"get_alias_funcs##module_name: {module_name}##alias: {alias_name}")

        module_name = module_name.strip()

        # 如果当前的api列表没有这个模块的api则要引入这个模块，引入这个模块才能用dir函数收集这个模块的方法列表
        try:
            imported_module = importlib.import_module(module_name)
        # 引入失败则安装这个模块
        except Exception as err:
            # 有"."号如models.bert说明需要安装根模块models，只需要安装根模块即可
            logger.debug(f"第一次引入模块失败，尝试安装{module_name}模块##{err}")
            try:
                if '.' in module_name:
                    index = module_name.find('.')
                    root_module = module_name[:index]
                    os.system('pip3 install ' + root_module)
                else:
                    os.system('pip3 install ' + module_name)
                imported_module = importlib.import_module(module_name)
            except Exception as err:
                logger.debug(f"安装{module_name}模块失败##{err}##返回空列表")
                return {}

        module_apis = dir(imported_module)  # dir函数返回这个模块的方法，属性列表

        return {alias_name: module_apis}

    # 获取module_name里的某个部分item_name（可能是类）方法列表
    @classmethod
    def get_item_funcs(cls, module_name, item_name):

        logger.debug(f"get_item_funcs##module_name: {module_name}##item_name: {item_name}")

        module_name = module_name.strip()

        try:
            imported_module = importlib.import_module(module_name)
        except Exception as err:
            logger.debug(f"第一次引入模块失败，尝试安装{module_name}模块##{err}")
            try:
                if '.' in module_name:
                    index = module_name.find('.')
                    root_module = module_name[:index]
                    os.system('pip3 install ' + root_module)
                else:
                    os.system('pip3 install ' + module_name)
                imported_module = importlib.import_module(module_name)
            except Exception as err:
                logger.debug(f"安装{module_name}模块失败##{err}##返回空列表")
                return {}

        try:
            item = getattr(imported_module, item_name)
            return {item_name: dir(item)}
        except Exception as err:
            logger.debug(f"获取{module_name}的{item_name}属性失败，尝试引入{module_name}.{item_name}##{err}")
            try:
                sub_module = importlib.import_module(module_name + '.' + item_name)
                return {item_name: dir(sub_module)}
            except Exception as err:
                logger.debug(f"引入{module_name}.{item_name}失败，返回空列表##{err}")
                return {}

    # 获取modulename里的某个部分itname并起别名（类或方法）的方法列表
    @classmethod
    def get_item_alias_funcs(cls, module_name, item_name, alias_name):

        logger.debug(
            f"get_item_alias_funcs##module_name: {module_name}##item_name: {item_name}##alias_name: {alias_name}")

        module_name = module_name.strip()

        try:
            imported_module = importlib.import_module(module_name)
        except Exception as err:
            logger.debug(f"第一次引入模块失败，尝试安装{module_name}模块##{err}")
            try:
                if '.' in module_name:
                    index = module_name.find('.')
                    root_module = module_name[:index]
                    os.system('pip3 install ' + root_module)
                else:
                    os.system('pip3 install ' + module_name)
                imported_module = importlib.import_module(module_name)
            except Exception as err:
                logger.debug(f"安装{module_name}模块失败##{err}##返回空列表")
                return {}

        try:
            item = getattr(imported_module, item_name)
            return {alias_name: dir(item)}
        except Exception as err:
            logger.debug(f"获取{module_name}的{item}属性失败，尝试引入{module_name}.{item}##{err}")
            try:
                sub_module = importlib.import_module(module_name + '.' + item_name)
                return {alias_name: dir(sub_module)}
            except Exception as err:
                logger.debug(f"引入{module_name}.{item}失败，返回空列表##{err}")
                return {}

    @classmethod
    def get_file_module_name(cls, file_path: Path, data_dir: Path):

        module_path = file_path.relative_to(data_dir).with_suffix("")
        module_name = str(module_path).replace(os.path.sep, ".")
        # module_path = file_path.split(raw_data_path + os.sep)[-1]
        # module_name = module_path.replace(os.path.sep, ".")[:-3]

        return module_name

    @classmethod
    def get_local_module(cls, file_path: Path, data_dir, module_name, level):

        module_path = file_path.relative_to(data_dir)
        for _ in range(level):
            local_module_path = os.path.dirname(module_path)
        if module_name is None:
            local_module = local_module_path.replace(os.path.sep, ".")
        else:
            local_module = local_module_path.replace(os.path.sep, ".") + "." + module_name

        return local_module

    # 获取typeshed的api，typeshed.txt文件路径可能要更改一下
    @staticmethod
    def get_typeshed_apis(caller_type):
        ret_apis = []
        caller_type = caller_type.strip()
        caller_type = re.sub(r'\[.*\]', '', caller_type)
        with open(config["other"]["typeshed_path"]) as f:
            lines = f.readlines()
        s1 = '.' + caller_type + '.'
        s2 = caller_type + '.'
        for line in lines:
            if s1 in line or line.startswith(s2):
                s3 = line.strip()
                index = s3.rfind('.')
                s4 = s3[index + 1:]
                if not s4 in ret_apis:
                    ret_apis.append(s4)
        return ret_apis
