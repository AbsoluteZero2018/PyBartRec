# 这是一个示例 Python 脚本。

# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。
import sys
import os
import subprocess
import config
from tqdm import tqdm
import torch
import math
import logging

from transformers import get_scheduler, SchedulerType
from accelerate import Accelerator
from data.code_rec_dataset import CodeRecDataset
from data.tools.data.vocab import Vocab, load_vocab
from data.code_rec_dataset import CodeRecDataset, collate_fn
from torch.utils.data.dataloader import DataLoader
from model.rec_spt_model import RecSPTModel

logger = logging.getLogger(__name__)


def print_hi(name):
    # 在下面的代码行中使用断点来调试脚本。
    print(f'Hi, {name}')  # 按 Ctrl+F8 切换断点。


def init_config():
    print(os.getcwd())
    os.system("export PYTHONPATH=" + os.getcwd())


def test_spt_model():
    trained_vocab = "/home/jeremy/Works/Python-Projects/SPT-Code-main/pre_trained/mod_vocabs"
    trained_model = "/home/jeremy/Works/Python-Projects/SPT-Code-main/pre_trained/models/all"
    # --------------------------------------------------
    # datasets
    # --------------------------------------------------
    batch_size = 12
    train_dataset = CodeRecDataset(split="train")
    test_dataset = CodeRecDataset(split="test")
    test_rec_info = test_dataset.test_rec_info
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                                  collate_fn=lambda batch: collate_fn(
                                      batch,
                                      code_vocab=code_vocab,
                                      nl_vocab=nl_vocab,
                                      ast_vocab=ast_vocab
                                  ))
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size,
                                 collate_fn=lambda batch: collate_fn(
                                     batch,
                                     code_vocab=code_vocab,
                                     nl_vocab=nl_vocab,
                                     ast_vocab=ast_vocab
                                 ))

    # --------------------------------------------------
    # vocabs
    # --------------------------------------------------
    code_vocab = load_vocab(vocab_root=trained_vocab, name="code")
    ast_vocab = load_vocab(vocab_root=trained_vocab, name="ast")
    nl_vocab = load_vocab(vocab_root=trained_vocab, name="nl")

    # --------------------------------------------------
    # model
    # --------------------------------------------------
    rec_spt_model = RecSPTModel(trained_model=trained_model)
    # optimizer
    learning_rate = 5e-5
    epochs = 1
    warmup_steps = 1000
    optimizer = torch.optim.Adam(rec_spt_model.parameters(), lr=learning_rate)
    grad_clipping_norm = 1.0

    accelerator = Accelerator()
    print(accelerator.device)
    rec_spt_model, optimizer, train_dataloader = accelerator.prepare(rec_spt_model, optimizer, train_dataloader)
    gradient_accumulation_steps = 1  # 多少个批后进行梯度更新
    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)  # 一轮进行多少次梯度更新
    max_train_steps = epochs * num_update_steps_per_epoch  # 梯度更新的最大次数

    lr_scheduler = get_scheduler(name=SchedulerType.LINEAR,
                                 optimizer=optimizer,
                                 num_warmup_steps=warmup_steps,
                                 num_training_steps=max_train_steps)

    total_batch_size = batch_size * accelerator.num_processes * gradient_accumulation_steps  # 实际喂给模型的批大小

    progress_bar = tqdm(range(max_train_steps))
    completed_steps = 0

    rec_spt_model.train()
    for epoch in range(epochs):
        # 一次epoch训练step步，每次取batch数据训练
        for step, batch in enumerate(train_dataloader):
            loss = rec_spt_model(**batch)[0]  # 每一次前向传播计算loss
            loss = loss / gradient_accumulation_steps
            accelerator.backward(loss)

            # if step % args.log_state_every == 0 and step != 0:
            if step % 10 == 0 and step != 0:
                logger.info('loss: {:.4f}'.format(loss.item()))

            if step % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                if grad_clipping_norm is not None and grad_clipping_norm > 0:

                    if hasattr(optimizer, "clip_grad_norm"):
                        # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
                        optimizer.clip_grad_norm(grad_clipping_norm)
                    elif hasattr(rec_spt_model, "clip_grad_norm_"):
                        # Some models (like FullyShardedDDP) have a specific way to do gradient clipping
                        rec_spt_model.clip_grad_norm_(grad_clipping_norm)
                    else:
                        # Revert to normal clipping otherwise, handling Apex or full precision
                        torch.nn.utils.clip_grad_norm_(rec_spt_model.parameters(), grad_clipping_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

            if completed_steps >= max_train_steps:
                break


# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    init_config()
    test_spt_model()

# 访问 https://www.jetbrains.com/help/pycharm/ 获取 PyCharm 帮助
