import torch
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn.functional as f

from transformers import BartForConditionalGeneration, BartConfig
from transformers.models.bart.modeling_bart import BartClassificationHead, shift_tokens_right
from transformers.modeling_outputs import Seq2SeqLMOutput, Seq2SeqSequenceClassifierOutput

from tqdm import tqdm
import numpy as np
import logging

import enums
from .utils import inputs_to_cuda

logger = logging.getLogger(__name__)


class BartForClassificationAndGeneration(BartForConditionalGeneration):

    def __init__(self, config: BartConfig, mode=None):
        # config全都用来初始化父类了，然后config最终用来初始化BartModel
        super(BartForClassificationAndGeneration, self).__init__(config)
        self.mode = None
        if mode:
            self.set_model_mode(mode)

        # classification head  用于分类部分的层，位于整个Bart模型的上面
        self.classification_head = BartClassificationHead(
            config.d_model,  # 输入维度
            config.d_model,  # 内层维度
            config.num_labels,  # 类别数
            config.classifier_dropout,
        )
        # 父类中定义的self.model
        self.model._init_weights(self.classification_head.dense)
        self.model._init_weights(self.classification_head.out_proj)

    def set_model_mode(self, mode):
        assert mode in [enums.MODEL_MODE_GEN, enums.MODEL_MODE_CLS, enums.MODEL_MODE_SEARCH]
        self.mode = mode
        logging.info(f'BART mode switched to {mode}')

    # 根据不同的任务进行不同的前向传播
    def forward(
            self,
            input_ids=None,  # 词表中输入序列token的索引，默认情况下忽略填充，除非提供padding。
            attention_mask=None,  # 注意力mask，避免注意到padding这些特殊token。
            decoder_input_ids=None,  # 词表中decoder输入序列token的索引，这些在collate_fn中创建了
            # Bart使用eos_token_id作为decoder_input_ids生成的起始令牌，如果使用past_key_values，可以选择只输入
            # 最后一个decoder_input_ids。对于翻译和摘要训练，应该提供decoder_input_ids。如果没有提供decoder_input_ids，
            # 模型将通过向右移动input_ids来创建这个张量，以便依据论文对预训练进行去噪。
            decoder_attention_mask=None,
            head_mask=None,  # 屏蔽编码器中某个注意力头的输出
            decoder_head_mask=None,
            cross_attn_head_mask=None,  # 屏蔽解码器中某个交叉注意力头的输出
            encoder_outputs=None,  # 编码器最后一层输出处的hidden_state序列，给解码器的交叉注意力使用
            past_key_values=None,  # 包含预先计算的隐藏状态（K和V矩阵），用于加快顺序解码
            inputs_embeds=None,  # 可以选择不输入input_id，而是直接输入input_id在embedding后的向量
            decoder_inputs_embeds=None,
            labels=None,
            use_cache=None,  # 使用后可以返回past_key_values的K和V矩阵，可以加快decoder解码
            output_attentions=None,  # 是否返回所有注意力层的注意力张量
            output_hidden_states=None,  # 是否返回所有层的隐藏状态
            return_dict=None,  # 是否返回ModelOutput而不是一个张量元组
            neg_nl_input_ids=None,
            neg_nl_attention_mask=None
    ):
        assert self.mode, 'It is required to specific a mode for BART before the model is passed through'

        # 只有后面两个参数跟其他mode不一样
        if self.mode == enums.MODEL_MODE_SEARCH:
            return self.forward_search(input_ids=input_ids,
                                       attention_mask=attention_mask,
                                       decoder_input_ids=decoder_input_ids,
                                       decoder_attention_mask=decoder_attention_mask,
                                       head_mask=head_mask,
                                       decoder_head_mask=decoder_head_mask,
                                       cross_attn_head_mask=cross_attn_head_mask,
                                       encoder_outputs=encoder_outputs,
                                       past_key_values=past_key_values,
                                       inputs_embeds=inputs_embeds,
                                       decoder_inputs_embeds=decoder_inputs_embeds,
                                       labels=labels,
                                       use_cache=use_cache,
                                       output_attentions=output_attentions,
                                       output_hidden_states=output_hidden_states,
                                       return_dict=return_dict,
                                       neg_nl_input_ids=neg_nl_input_ids,
                                       neg_nl_attention_mask=neg_nl_attention_mask)

        elif self.mode == enums.MODEL_MODE_GEN:

            return self.forward_gen(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    decoder_input_ids=decoder_input_ids,
                                    decoder_attention_mask=decoder_attention_mask,
                                    head_mask=head_mask,
                                    decoder_head_mask=decoder_head_mask,
                                    cross_attn_head_mask=cross_attn_head_mask,
                                    encoder_outputs=encoder_outputs,
                                    past_key_values=past_key_values,
                                    inputs_embeds=inputs_embeds,
                                    decoder_inputs_embeds=decoder_inputs_embeds,
                                    labels=labels,
                                    use_cache=use_cache,
                                    output_attentions=output_attentions,
                                    output_hidden_states=output_hidden_states,
                                    return_dict=return_dict)

        elif self.mode == enums.MODEL_MODE_CLS:
            return self.forward_cls(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    decoder_input_ids=decoder_input_ids,
                                    decoder_attention_mask=decoder_attention_mask,
                                    head_mask=head_mask,
                                    decoder_head_mask=decoder_head_mask,
                                    cross_attn_head_mask=cross_attn_head_mask,
                                    encoder_outputs=encoder_outputs,
                                    past_key_values=past_key_values,
                                    inputs_embeds=inputs_embeds,
                                    decoder_inputs_embeds=decoder_inputs_embeds,
                                    labels=labels,
                                    use_cache=use_cache,
                                    output_attentions=output_attentions,
                                    output_hidden_states=output_hidden_states,
                                    return_dict=return_dict)

    def forward_gen(
            self,
            input_ids=None,  # 词表中输入序列token的索引，默认情况下忽略填充，除非提供padding。
            attention_mask=None,  # 注意力mask，避免注意到padding这些特殊token。
            decoder_input_ids=None,  # 词表中decoder输入序列token的索引
            decoder_attention_mask=None,  # Bart使用eos_token_id作为decoder_input_ids生成的起始令牌，如果使用past_key_values，可以选择只输入最后一个decoder_input_ids
            head_mask=None,
            decoder_head_mask=None,
            cross_attn_head_mask=None,
            encoder_outputs=None,
            past_key_values=None,
            inputs_embeds=None,
            decoder_inputs_embeds=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None
    ):
        # self.config继承自PreTrainedModel父类
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict  # 是否返回ModelOutput数据类

        if labels is not None:
            if decoder_input_ids is None:
                # 这是文本生成任务，shift_tokens_right方法是用来将decoder层的输入向后移动一个位置
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        # self.model是可调用对象，__call__方法在Pytorch中的Module模型中定义，调用此方法会调用模型内的forward函数，返回的应该是传入下一
        # 层的张量
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # lm_head应该是一个输出层的张量
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias  # lm_logits也是一个张量，一般为最终的全连接层的输出

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))  # 也是一个张量

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,  # 如果labels提供，则返回损失
            # 在softmax层语言模型头对每个词表token的预测得分
            logits=lm_logits,  # shape `(batch_size, sequence_length, config.vocab_size)`
            # 包含预先计算的hidden_state（自注意力块和交叉注意力块中的K和V矩阵）
            past_key_values=outputs.past_key_values,  # 防止模型在文本生成任务中重新计算上一次迭代中已经计算好的上下文的值
            decoder_hidden_states=outputs.decoder_hidden_states,  # 一个嵌入输出和一个每层输出的元组，解码器在每层输出处的隐藏状态加上初始嵌入输出。
            decoder_attentions=outputs.decoder_attentions,  # 经过注意力的softmax层后，解码器的注意力权重，用于计算自注意头的加权平均值。
            cross_attentions=outputs.cross_attentions,  # 经过注意力的softmax层后，解码器的交叉注意力层的权重，用于计算交叉注意头的加权平均值。
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,  # 模型encoder的最后一层输出的hidden_state序列
            encoder_hidden_states=outputs.encoder_hidden_states,  # 编码器在每层输出处的隐藏状态加上初始嵌入输出。
            encoder_attentions=outputs.encoder_attentions,  # 经过注意力的softmax层后，编码器的注意力权重，用于计算自注意头的加权平均值。
        )

    def forward_representation(
            self,
            input_ids=None,
            attention_mask=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            cross_attn_head_mask=None,
            encoder_outputs=None,
            past_key_values=None,
            inputs_embeds=None,
            decoder_inputs_embeds=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None):

        # 利用bart模型来做向量编码
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]  # last hidden state

        eos_mask = input_ids.eq(self.config.eos_token_id)

        if len(torch.unique(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")
        # view方法相当于reshape，重新定义矩阵的形状
        sentence_representation = hidden_states[eos_mask, :].view(hidden_states.size(0), -1,
                                                                  hidden_states.size(-1))[
                                  :, -1, :
                                  ]
        print(sentence_representation)
        print(type(outputs))
        return sentence_representation, outputs

    def forward_cls(
            self,
            input_ids=None,
            attention_mask=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            cross_attn_head_mask=None,
            encoder_outputs=None,
            past_key_values=None,
            inputs_embeds=None,
            decoder_inputs_embeds=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if labels is not None:
            use_cache = False

        if input_ids is None and inputs_embeds is not None:
            raise NotImplementedError(
                f"Passing input embeddings is currently not supported for {self.__class__.__name__}"
            )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]  # last hidden state

        eos_mask = input_ids.eq(self.config.eos_token_id)

        if len(torch.unique(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")
        sentence_representation = hidden_states[eos_mask, :].view(hidden_states.size(0), -1,
                                                                  hidden_states.size(-1))[
                                  :, -1, :
                                  ]
        logits = self.classification_head(sentence_representation)

        loss = None
        if labels is not None:
            if self.config.num_labels == 1:
                # regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    def forward_search(
            self,
            input_ids=None,
            attention_mask=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            cross_attn_head_mask=None,
            encoder_outputs=None,
            past_key_values=None,
            inputs_embeds=None,
            decoder_inputs_embeds=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            neg_nl_input_ids=None,
            neg_nl_attention_mask=None
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        code_representation, code_outputs = self.forward_representation(input_ids=input_ids,
                                                                        attention_mask=attention_mask,
                                                                        # decoder_input_ids=None,
                                                                        # decoder_attention_mask=decoder_attention_mask,
                                                                        # head_mask=head_mask,
                                                                        # decoder_head_mask=decoder_head_mask,
                                                                        # cross_attn_head_mask=cross_attn_head_mask,
                                                                        # encoder_outputs=encoder_outputs,
                                                                        # past_key_values=past_key_values,
                                                                        # inputs_embeds=inputs_embeds,
                                                                        # decoder_inputs_embeds=None,
                                                                        # labels=None,
                                                                        use_cache=use_cache,
                                                                        # output_attentions=output_attentions,
                                                                        # output_hidden_states=output_hidden_states,
                                                                        return_dict=return_dict)
        nl_representation, nl_outputs = self.forward_representation(input_ids=decoder_input_ids,
                                                                    attention_mask=decoder_attention_mask,
                                                                    # decoder_input_ids=None,
                                                                    # decoder_attention_mask=None,
                                                                    # head_mask=head_mask,
                                                                    # decoder_head_mask=decoder_head_mask,
                                                                    # cross_attn_head_mask=cross_attn_head_mask,
                                                                    # encoder_outputs=encoder_outputs,
                                                                    # past_key_values=past_key_values,
                                                                    # inputs_embeds=inputs_embeds,
                                                                    # decoder_inputs_embeds=None,
                                                                    # labels=None,
                                                                    use_cache=use_cache,
                                                                    # output_attentions=output_attentions,
                                                                    # output_hidden_states=output_hidden_states,
                                                                    return_dict=return_dict)

        neg_nl_representation, neg_nl_outputs = self.forward_representation(input_ids=neg_nl_input_ids,
                                                                            attention_mask=neg_nl_attention_mask,
                                                                            use_cache=use_cache,
                                                                            return_dict=return_dict)

        pos_sim = f.cosine_similarity(code_representation, nl_representation)
        neg_sim = f.cosine_similarity(code_representation, neg_nl_representation)

        loss = (0.413 - pos_sim + neg_sim).clamp(min=1e-6).mean()
        return loss

        # if not return_dict:
        #     output = (code_representation,) + code_outputs[1:]
        #     return ((loss,) + output) if loss is not None else output
        #
        # return Seq2SeqSequenceClassifierOutput(
        #     loss=loss,
        #     logits=code_representation,
        #     past_key_values=code_outputs.past_key_values,
        #     decoder_hidden_states=code_outputs.decoder_hidden_states,
        #     decoder_attentions=code_outputs.decoder_attentions,
        #     cross_attentions=code_outputs.cross_attentions,
        #     encoder_last_hidden_state=code_outputs.encoder_last_hidden_state,
        #     encoder_hidden_states=code_outputs.encoder_hidden_states,
        #     encoder_attentions=code_outputs.encoder_attentions,
        # )

    # 先将文档字符串编码向量，然后将候选代码编码为向量，计算两者之间的相似度并形成排名列表，然后将排名列表跟正例代码比较，找到正例代码的排名
    def evaluate_search(self,
                        query_dataloader: torch.utils.data.dataloader.DataLoader,
                        codebase_dataloader: torch.utils.data.dataloader.DataLoader,
                        metrics_prefix: str):

        self.set_model_mode(enums.MODEL_MODE_CLS)
        self.eval()  # 设置模型为评估模式

        # embed query and codebase
        with torch.no_grad():  # 在该模块下，所有计算得出的tensor的requires_grad都自动设置为False，反向传播不会自动求导
            logger.info('(1/3) Embedding search queries')
            query_vectors = []  # 查询的文档
            ref_urls = []  # 对应的正例代码
            for _, batch in enumerate(tqdm(query_dataloader)):
                urls = batch.pop('urls')
                ref_urls += urls  # 所有的正例代码？
                batch = inputs_to_cuda(batch)
                representation, outputs = self.forward_representation(**batch)  # representation: [B, H]
                representation = representation.cpu().numpy()  # [B, H]
                query_vectors.append(representation)
            query_vectors = np.vstack(query_vectors)  # [len_query, H]

            logger.info('(2/3) Embedding candidate codes')  # 候选的codes就是在codebase里的全部url
            code_vectors = []  # 变成向量后的正例代码
            code_urls = []  # 原始的候选代码
            for _, batch in enumerate(tqdm(codebase_dataloader)):
                urls = batch.pop('urls')
                code_urls += urls
                batch = inputs_to_cuda(batch)
                representation, outputs = self.forward_representation(**batch)
                representation = representation.cpu().numpy()
                code_vectors.append(representation)
            code_vectors = np.vstack(code_vectors)  # [len_code, H]

            # calculate MRR
            logger.info('(3/3) Calculating metrics')
            scores = []
            ranks = []
            can_urls = []
            can_sims = []
            for query_vector, ref_url in tqdm(zip(query_vectors, ref_urls), total=len(query_vectors)):
                sims = []
                for code_vector, code_url in zip(code_vectors, code_urls):
                    # 计算余弦相似度
                    sim = f.cosine_similarity(torch.from_numpy(code_vector).unsqueeze(0),
                                              torch.from_numpy(query_vector).unsqueeze(0))
                    sims.append((code_url, sim.item()))
                sims.sort(key=lambda item: item[1])

                sims = sims[:1000]
                can_urls.append(sims[0][0])  # 没有编码为向量前的候选代码
                can_sims.append(sims[0][1])  # 对应的相似度

                rank = -1
                for index, (url, sim) in enumerate(sims):
                    if url == ref_url:
                        rank = index + 1
                ranks.append(rank)  # 推荐出正确代码的排名
                score = 1 / rank if rank != -1 else 0
                scores.append(score)

        self.train()
        self.set_model_mode(enums.MODEL_MODE_SEARCH)

        results = {f'{metrics_prefix}_mrr': np.mean(scores),
                   f'{metrics_prefix}_ranks': ranks,
                   f'{metrics_prefix}_ref_urls': ref_urls,
                   f'{metrics_prefix}_can_urls': can_urls,
                   f'{metrics_prefix}_can_sims': can_sims}
        return results
