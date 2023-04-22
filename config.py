# _*_ coding: utf-8 _*_
# @Time    :   2023/01/05 18:11:04
# @FileName:   config.py
# @Author  :   423A35C7
# @Software:   VSCode

import os
import sys
from pathlib import Path
from configobj import ConfigObj

# config直接在config所在目录运行
config_path = Path(os.path.join(os.getcwd(), "config.ini")).expanduser()
if not config_path.parent.is_dir():
    config_path.parent.mkdir(parents=True, exist_ok=True)
config = ConfigObj(str(config_path), encoding='utf-8')
config.clear()  # 测试用，测试完了就去掉

config.setdefault("datasets", {})
config["datasets"].setdefault("raw-data", os.path.abspath("data/datasets/raw-data"))
config["datasets"].setdefault("clean-data", os.path.abspath("data/datasets/clean-data"))
config["datasets"].setdefault("json-data", os.path.abspath("data/datasets/json-data"))
config["datasets"].setdefault("encoded-data", os.path.abspath("data/datasets/encoded-data"))
config["datasets"].setdefault("log", os.path.abspath("data/logs"))

config.setdefault("project", {})  # {}代表项目名称
config["project"].setdefault("proj_name", "flask")
config["project"].setdefault("proj_json", os.path.abspath("data/files/{}.json"))
config["project"].setdefault("proj_train_json_list", os.path.abspath("data/datasets/encoded-data/{}_train.jsonl"))
config["project"].setdefault("proj_test_json_list", os.path.abspath("data/datasets/encoded-data/{}_test.jsonl"))
config["project"].setdefault("proj_train_json_list_files", os.path.abspath("data/datasets/encoded-data/{}_train_files.txt"))
config["project"].setdefault("proj_test_json_list_files", os.path.abspath("data/datasets/encoded-data/{}_test_files.txt"))
config["project"].setdefault("token_count", os.path.abspath("data/datasets/json-data/{}_token_count.json"))
config["project"].setdefault("depends", os.path.abspath("data/datasets/json-data/{}_depends.json"))
config["project"].setdefault("token_no", os.path.abspath("data/datasets/json-data/{}_token_no.json"))
config["project"].setdefault("proj_feature_data", os.path.abspath("data/datasets/encoded-data/{}_feature_data.csv"))
config["project"].setdefault("proj_feature_label", os.path.abspath("data/datasets/encoded-data/{}_feature_label.csv"))

config.setdefault("log", {})
config["log"].setdefault("info", os.path.abspath("data/logs/info.log"))
config["log"].setdefault("feature_encoder", os.path.abspath("data/logs/feature_encoder.log"))
config["log"].setdefault("data_utils", os.path.abspath("data/logs/data_utils.log"))
config["log"].setdefault("rec_rfc_model", os.path.abspath("model/logs/rec_rfc_model.log"))

config.setdefault("other", {})
config["other"].setdefault("builtin_path", os.path.abspath("data/files/builtin.txt"))
config["other"].setdefault("typeshed_path", os.path.abspath("data/files/typeshed.txt"))
config["other"].setdefault("trainfile_path", os.path.abspath("data/files/trainfile.lm"))
config["other"].setdefault("runtime_pip", f"{sys.executable} -m pip")
config["other"].setdefault("ngram_path", "/home/jeremy/Works/Python-Projects/PyBartRec/data/tools/srilm-1.7.2/lm/bin/i686-m64/ngram")

config.setdefault("model", {})
config["model"].setdefault("model_save_path", os.path.abspath("model/saved-model"))
config["model"].setdefault("result_save_path", os.path.abspath("model/result"))

config.write()