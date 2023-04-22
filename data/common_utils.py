# -*- coding=utf-8 -*-
# @Time: 2022/12/12 11:49
# @Author: Jeremy
# @File: common_utils.py
# @Software: PyCharm
import re


class CommonUtils:
    def __init__(self):
        pass

    @classmethod
    def get_blank(cls, code_line):
        blank_index = 0
        for blank_index in range(0, len(code_line)):
            if code_line[blank_index] != ' ':
                break
        return code_line[:blank_index], blank_index

    # 获取指定字符串的由开头和结尾字符串中间的字符串
    @classmethod
    def get_middle_str(cls, content, start_str, end_str):
        start_index = content.index(start_str)
        # 如果找到开始字符串位置
        if start_index >= 0:
            start_index += len(start_str)
        end_index = content.index(end_str)
        return content[start_index:end_index]


