# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch RoBERTa model. """


import logging

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn.functional as F
from torch.nn.utils import rnn
from torch.autograd import Variable
import numpy as np

from .configuration_adapter_roberta import AdapterRobertaConfig
from .modeling_utils_shared_hgn import PreTrainedModel, prune_linear_layer
from .hgn_graph_model import HierarchicalGraphNetwork

from collections import OrderedDict, UserDict
from collections.abc import MutableMapping
from contextlib import ExitStack
from enum import Enum
from typing import Any, ContextManager, List, Tuple
from typing import Optional, Tuple

from .file_utils import add_start_docstrings
from .modeling_adapter_bert_shared_gnn import BertEmbeddings, BertLayerNorm, BertModel, BertPreTrainedModel, gelu



logger = logging.getLogger(__name__)

ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP = {
    "roberta-base": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-pytorch_model.bin",
    "roberta-large": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-pytorch_model.bin",
    "roberta-large-mnli": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-mnli-pytorch_model.bin",
    "distilroberta-base": "https://s3.amazonaws.com/models.huggingface.co/bert/distilroberta-base-pytorch_model.bin",
    "roberta-base-openai-detector": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-openai-detector-pytorch_model.bin",
    "roberta-large-openai-detector": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-openai-detector-pytorch_model.bin",
}


class RobertaEmbeddings(BertEmbeddings):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """

    def __init__(self, config):
        super(RobertaEmbeddings, self).__init__(config)
        self.padding_idx = 1
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=self.padding_idx)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
        )

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
        if position_ids is None:
            if input_ids is not None:
                # Create the position ids from the input token ids. Any padded tokens remain padded.
                position_ids = self.create_position_ids_from_input_ids(input_ids).to(input_ids.device)
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

        return super(RobertaEmbeddings, self).forward(
            input_ids, token_type_ids=token_type_ids, position_ids=position_ids, inputs_embeds=inputs_embeds
        )

    def create_position_ids_from_input_ids(self, x):
        """ Replace non-padding symbols with their position numbers. Position numbers begin at
        padding_idx+1. Padding symbols are ignored. This is modified from fairseq's
        `utils.make_positions`.

        :param torch.Tensor x:
        :return torch.Tensor:
        """
        mask = x.ne(self.padding_idx).long()
        incremental_indicies = torch.cumsum(mask, dim=1) * mask
        return incremental_indicies + self.padding_idx

    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        """ We are provided embeddings directly. We cannot infer which are padded so just generate
        sequential position ids.

        :param torch.Tensor inputs_embeds:
        :return torch.Tensor:
        """
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]

        position_ids = torch.arange(
            self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
        )
        return position_ids.unsqueeze(0).expand(input_shape)


ROBERTA_START_DOCSTRING = r"""    The RoBERTa model was proposed in
    `RoBERTa: A Robustly Optimized BERT Pretraining Approach`_
    by Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer,
    Veselin Stoyanov. It is based on Google's BERT model released in 2018.

    It builds on BERT and modifies key hyperparameters, removing the next-sentence pretraining
    objective and training with much larger mini-batches and learning rates.

    This implementation is the same as BertModel with a tiny embeddings tweak as well as a setup for Roberta pretrained
    models.

    This model is a PyTorch `torch.nn.Module`_ sub-class. Use it as a regular PyTorch Module and
    refer to the PyTorch documentation for all matter related to general usage and behavior.

    .. _`RoBERTa: A Robustly Optimized BERT Pretraining Approach`:
        https://arxiv.org/abs/1907.11692

    .. _`torch.nn.Module`:
        https://pytorch.org/docs/stable/nn.html#module

    Parameters:
        config (:class:`~transformers.RobertaConfig`): Model configuration class with all the parameters of the
            model. Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.
"""




@add_start_docstrings(
    "The bare RoBERTa Model transformer outputting raw hidden-states without any specific head on top.",
    ROBERTA_START_DOCSTRING,
)
class RobertaModel(BertModel):
    r"""
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **last_hidden_state**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, hidden_size)``
            Sequence of hidden-states at the output of the last layer of the model.
        **pooler_output**: ``torch.FloatTensor`` of shape ``(batch_size, hidden_size)``
            Last layer hidden-state of the first token of the sequence (classification token)
            further processed by a Linear layer and a Tanh activation function. The Linear
            layer weights are trained from the next sentence prediction (classification)
            objective during Bert pretraining. This output is usually *not* a good summary
            of the semantic content of the input, you're often better with averaging or pooling
            the sequence of hidden-states for the whole input sequence.
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaModel.from_pretrained('roberta-base')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)
        last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple

    """
    config_class = AdapterRobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config, hgn):
        super(RobertaModel, self).__init__(config, hgn)

        self.embeddings = RobertaEmbeddings(config)
        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value


@add_start_docstrings(
    """RoBERTa Model with a `language modeling` head on top. """, ROBERTA_START_DOCSTRING
)
class RobertaForMaskedLM(BertPreTrainedModel):
    r"""
        **masked_lm_labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Labels for computing the masked language modeling loss.
            Indices should be in ``[-100, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
            Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens with labels
            in ``[0, ..., config.vocab_size]``

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``masked_lm_labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Masked language modeling loss.
        **prediction_scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, config.vocab_size)``
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaForMaskedLM.from_pretrained('roberta-base')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, masked_lm_labels=input_ids)
        loss, prediction_scores = outputs[:2]

    """
    config_class = AdapterRobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super(RobertaForMaskedLM, self).__init__(config)

        self.roberta = RobertaModel(config)
        self.lm_head = RobertaLMHead(config)

        self.init_weights()

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        masked_lm_labels=None,
    ):
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)

        outputs = (prediction_scores,) + outputs[2:]  # Add hidden states and attention if they are here

        if masked_lm_labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            outputs = (masked_lm_loss,) + outputs

        return outputs  # (masked_lm_loss), prediction_scores, (hidden_states), (attentions)


class RobertaLMHead(nn.Module):
    """Roberta Head for masked language modeling."""

    def __init__(self, config):
        super(RobertaLMHead, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)

        # project back to size of vocabulary with bias
        x = self.decoder(x) + self.bias

        return x


@add_start_docstrings(
    """RoBERTa Model transformer with a sequence classification/regression head on top (a linear layer
    on top of the pooled output) e.g. for GLUE tasks. """,
    ROBERTA_START_DOCSTRING,
)
class RobertaForSequenceClassification(BertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the sequence classification/regression loss.
            Indices should be in ``[0, ..., config.num_labels]``.
            If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification (or regression if config.num_labels==1) loss.
        **logits**: ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaForSequenceClassification.from_pretrained('roberta-base')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]

    """
    config_class = AdapterRobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super(RobertaForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config)
        self.classifier = RobertaClassificationHead(config)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[2:]
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


@add_start_docstrings(
    """Roberta Model with a multiple choice classification head on top (a linear layer on top of
    the pooled output and a softmax) e.g. for RocStories/SWAG tasks. """,
    ROBERTA_START_DOCSTRING,
    
)
class RobertaForMultipleChoice(BertPreTrainedModel):
    r"""
    Inputs:
        **input_ids**: ``torch.LongTensor`` of shape ``(batch_size, num_choices, sequence_length)``:
            Indices of input sequence tokens in the vocabulary.
            The second dimension of the input (`num_choices`) indicates the number of choices to score.
            To match pre-training, RoBerta input sequence should be formatted with [CLS] and [SEP] tokens as follows:

            (a) For sequence pairs:

                ``tokens:         [CLS] is this jack ##son ##ville ? [SEP] [SEP] no it is not . [SEP]``

                ``token_type_ids:   0   0  0    0    0     0       0   0   0     1  1  1  1   1   1``

            (b) For single sequences:

                ``tokens:         [CLS] the dog is hairy . [SEP]``

                ``token_type_ids:   0   0   0   0  0     0   0``

            Indices can be obtained using :class:`transformers.BertTokenizer`.
            See :func:`transformers.PreTrainedTokenizer.encode` and
            :func:`transformers.PreTrainedTokenizer.convert_tokens_to_ids` for details.
        **token_type_ids**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, num_choices, sequence_length)``:
            Segment token indices to indicate first and second portions of the inputs.
            The second dimension of the input (`num_choices`) indicates the number of choices to score.
            Indices are selected in ``[0, 1]``: ``0`` corresponds to a `sentence A` token, ``1``
        **attention_mask**: (`optional`) ``torch.FloatTensor`` of shape ``(batch_size, num_choices, sequence_length)``:
            Mask to avoid performing attention on padding token indices.
            The second dimension of the input (`num_choices`) indicates the number of choices to score.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
        **head_mask**: (`optional`) ``torch.FloatTensor`` of shape ``(num_heads,)`` or ``(num_layers, num_heads)``:
            Mask to nullify selected heads of the self-attention modules.
            Mask values selected in ``[0, 1]``:
            ``1`` indicates the head is **not masked**, ``0`` indicates the head is **masked**.
        **inputs_embeds**: (`optional`) ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, embedding_dim)``:
            Optionally, instead of passing ``input_ids`` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert `input_ids` indices into associated vectors
            than the model's internal embedding lookup matrix.
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the multiple choice classification loss.
            Indices should be in ``[0, ..., num_choices]`` where `num_choices` is the size of the second dimension
            of the input tensors. (see `input_ids` above)

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification loss.
        **classification_scores**: ``torch.FloatTensor`` of shape ``(batch_size, num_choices)`` where `num_choices` is the size of the second dimension
            of the input tensors. (see `input_ids` above).
            Classification scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaForMultipleChoice.from_pretrained('roberta-base')
        choices = ["Hello, my dog is cute", "Hello, my cat is amazing"]
        input_ids = torch.tensor([tokenizer.encode(s, add_special_tokens=True) for s in choices]).unsqueeze(0)  # Batch size 1, 2 choices
        labels = torch.tensor(1).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, classification_scores = outputs[:2]

    """
    config_class = AdapterRobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super(RobertaForMultipleChoice, self).__init__(config)

        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        attention_mask=None,
        labels=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
    ):
        num_choices = input_ids.shape[1]

        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        outputs = self.roberta(
            flat_input_ids,
            position_ids=flat_position_ids,
            token_type_ids=flat_token_type_ids,
            attention_mask=flat_attention_mask,
            head_mask=head_mask,
        )
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        outputs = (reshaped_logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
            outputs = (loss,) + outputs

        return outputs  # (loss), reshaped_logits, (hidden_states), (attentions)


@add_start_docstrings(
    """Roberta Model with a token classification head on top (a linear layer on top of
    the hidden-states output) e.g. for Named-Entity-Recognition (NER) tasks. """,
    ROBERTA_START_DOCSTRING,
)
class RobertaForTokenClassification(BertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Labels for computing the token classification loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification loss.
        **scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, config.num_labels)``
            Classification scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaForTokenClassification.from_pretrained('roberta-base')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1] * input_ids.size(1)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, scores = outputs[:2]

    """
    config_class = AdapterRobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super(RobertaForTokenClassification, self).__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), scores, (hidden_states), (attentions)


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super(RobertaClassificationHead, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


@add_start_docstrings(
    """Roberta Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear layers on top of
    the hidden-states output to compute `span start logits` and `span end logits`). """,
    ROBERTA_START_DOCSTRING,
)
class RobertaForQuestionAnswering(BertPreTrainedModel):
    r"""
        **start_positions**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.
        **end_positions**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
        **start_scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length,)``
            Span-start scores (before SoftMax).
        **end_scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length,)``
            Span-end scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    Examples::
        tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        model = RobertaForQuestionAnswering.from_pretrained('roberta-large')
        question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
        input_ids = tokenizer.encode(question, text)
        start_scores, end_scores = model(torch.tensor([input_ids]))
        all_tokens = tokenizer.convert_ids_to_tokens(input_ids)
        answer = ' '.join(all_tokens[torch.argmax(start_scores) : torch.argmax(end_scores)+1])
    """
    config_class = AdapterRobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super(RobertaForQuestionAnswering, self).__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        start_positions=None,
        end_positions=None,
    ):

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        outputs = (start_logits, end_logits,) + outputs[2:]
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            outputs = (total_loss,) + outputs

        return outputs  # (loss), start_logits, end_logits, (hidden_states), (attentions)


class AdapterGraphQA(torch.nn.Module):
    def __init__(self, pretrained_weights, args):
        super().__init__()     

        # HGN
        self.hgn = HierarchicalGraphNetwork(args)
        
        # Transformer Encoder
        self.encoder = RobertaModel.from_pretrained(pretrained_weights, hgn=self.hgn, adapter_size=args.adapter_size)
        # hgn is shared across the layers of the transformer encoder
        self.hidden_size = self.encoder.config.hidden_size

        # Span Prediction
        q_dim = self.hidden_dim if args.q_update else args.input_dim
        self.predict_layer = PredictionLayer(args, q_dim)

    def forward(
            self,
            batch,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            return_yp=True,
        ):
            r"""
            start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
                Labels for position (index) of the start of the labelled span for computing the token classification loss.
                Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
                are not taken into account for computing the loss.
            end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
                Labels for position (index) of the end of the labelled span for computing the token classification loss.
                Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
                are not taken into account for computing the loss.
            """
            # 1) get transformer token embeddings
            (transformer_output, para_predictions, sent_predictions, sent_logits, ent_predictions) = self.encoder(
                batch,
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
            )
            predictions = self.predict_layer(batch, transformer_output[0], sent_logits[-1], packing_mask=batch['query_mapping'], return_yp=return_yp)

            if return_yp:
                start, end, q_type, yp1, yp2 = predictions
                return start, end, q_type, para_predictions[-1], sent_predictions[-1], ent_predictions[-1], yp1, yp2
            else:
                start, end, q_type = predictions
                return start, end, q_type, para_predictions[-1], sent_predictions[-1], ent_predictions[-1]



class HierarchicalGraphNetwork(nn.Module):
    """
    Packing Query Version
    """
    def __init__(self, config):
        super(HierarchicalGraphNetwork, self).__init__()
        self.config = config
        self.max_query_length = self.config.max_query_length

        self.bi_attention = BiAttention(input_dim=config.input_dim,
                                        memory_dim=config.input_dim,
                                        hid_dim=config.hidden_dim,
                                        dropout=config.bi_attn_drop)
        self.bi_attn_linear = nn.Linear(config.hidden_dim * 4, config.hidden_dim)

        self.hidden_dim = config.hidden_dim

        self.sent_lstm = LSTMWrapper(input_dim=config.hidden_dim,
                                     hidden_dim=config.hidden_dim,
                                     n_layer=1,
                                     dropout=config.lstm_drop)

        self.graph_blocks = nn.ModuleList()
        for _ in range(self.config.num_gnn_layers):
            self.graph_blocks.append(GraphBlock(self.config.q_attn, config))

        self.ctx_attention = GatedAttention(input_dim=config.hidden_dim*2,
                                            memory_dim=config.hidden_dim if config.q_update else config.hidden_dim*2,
                                            hid_dim=self.config.input_dim,
                                            dropout=config.bi_attn_drop,
                                            gate_method=self.config.ctx_attn)

        q_dim = self.hidden_dim if config.q_update else config.input_dim

    def forward(self, context_encoding, batch):
        query_mapping = batch['query_mapping']

        # extract query encoding
        trunc_query_mapping = query_mapping[:, :self.max_query_length].contiguous()
        trunc_query_state = (context_encoding * query_mapping.unsqueeze(2))[:, :self.max_query_length, :].contiguous()
        # bert encoding query vec
        query_vec = mean_pooling(trunc_query_state, trunc_query_mapping)

        attn_output, trunc_query_state = self.bi_attention(context_encoding,
                                                           trunc_query_state,
                                                           trunc_query_mapping)

        input_state = self.bi_attn_linear(attn_output) # N x L x d
        input_state = self.sent_lstm(input_state, batch['context_lens'])

        if self.config.q_update:
            query_vec = mean_pooling(trunc_query_state, trunc_query_mapping)

        para_logits, sent_logits = [], []
        para_predictions, sent_predictions, ent_predictions = [], [], []

        for l in range(self.config.num_gnn_layers):
            new_input_state, graph_state, graph_mask, sent_state, query_vec, para_logit, para_prediction, \
            sent_logit, sent_prediction, ent_logit = self.graph_blocks[l](batch, input_state, query_vec)

            para_logits.append(para_logit)
            sent_logits.append(sent_logit)
            para_predictions.append(para_prediction)
            sent_predictions.append(sent_prediction)
            ent_predictions.append(ent_logit)

        input_state, _ = self.ctx_attention(input_state, graph_state, graph_mask.squeeze(-1))
        return (input_state, para_predictions, sent_predictions, sent_logits, ent_predictions)


def tok_to_ent(tok2ent):
    if tok2ent == 'mean':
        return MeanPooling
    elif tok2ent == 'mean_max':
        return MeanMaxPooling
    else:
        raise NotImplementedError

class MLP(nn.Module):
    def __init__(self, input_sizes, dropout_prob=0.2, bias=False):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(1, len(input_sizes)):
            self.layers.append(nn.Linear(input_sizes[i - 1], input_sizes[i], bias=bias))
        self.norm_layers = nn.ModuleList()
        if len(input_sizes) > 2:
            for i in range(1, len(input_sizes) - 1):
                self.norm_layers.append(nn.LayerNorm(input_sizes[i]))
        self.drop_out = nn.Dropout(p=dropout_prob)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(self.drop_out(x))
            if i < len(self.layers) - 1:
                x = gelu(x)
                if len(self.norm_layers):
                    x = self.norm_layers[i](x)
        return x


def mean_pooling(input, mask):
    mean_pooled = input.sum(dim=1) / mask.sum(dim=1, keepdim=True)
    return mean_pooled


class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, doc_state, entity_mapping, entity_lens):
        entity_states = entity_mapping.unsqueeze(3) * doc_state.unsqueeze(1)  # N x E x L x d
        mean_pooled = torch.sum(entity_states, dim=2) / entity_lens.unsqueeze(2)
        return mean_pooled

class MeanMaxPooling(nn.Module):
    def __init__(self):
        super(MeanMaxPooling, self).__init__()

    def forward(self, doc_state, entity_mapping, entity_lens):
        """
        :param doc_state:  N x L x d
        :param entity_mapping:  N x E x L
        :param entity_lens:  N x E
        :return: N x E x 2d
        """
        entity_states = entity_mapping.unsqueeze(3) * doc_state.unsqueeze(1)  # N x E x L x d
        max_pooled = torch.max(entity_states, dim=2)[0]
        mean_pooled = torch.sum(entity_states, dim=2) / entity_lens.unsqueeze(2)
        output = torch.cat([max_pooled, mean_pooled], dim=2)  # N x E x 2d
        return output

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class GATSelfAttention(nn.Module):
    def __init__(self, in_dim, out_dim, config, q_attn=False, head_id=0):
        """ One head GAT """
        super(GATSelfAttention, self).__init__()
        self.config = config
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = self.config.gnn_drop
        self.q_attn = q_attn
        self.query_dim = in_dim
        self.n_type = self.config.num_edge_type

        self.head_id = head_id
        self.step = 0

        self.W_type = nn.ParameterList()
        self.a_type = nn.ParameterList()
        self.qattn_W1 = nn.ParameterList()
        self.qattn_W2 = nn.ParameterList()
        for i in range(self.n_type):
            self.W_type.append(get_weights((in_dim, out_dim)))
            self.a_type.append(get_weights((out_dim * 2, 1)))

            if self.q_attn:
                q_dim = self.config.hidden_dim if self.config.q_update else self.config.input_dim
                self.qattn_W1.append(get_weights((q_dim, out_dim * 2)))
                self.qattn_W2.append(get_weights((out_dim * 2, out_dim * 2)))

        self.act = get_act('lrelu:0.2')

    def forward(self, input_state, adj, node_mask=None, query_vec=None):
        zero_vec = torch.zeros_like(adj)
        scores = torch.zeros_like(adj)

        for i in range(self.n_type):
            h = torch.matmul(input_state, self.W_type[i])
            h = F.dropout(h, self.dropout, self.training)
            N, E, d = h.shape

            a_input = torch.cat([h.repeat(1, 1, E).view(N, E * E, -1), h.repeat(1, E, 1)], dim=-1)
            a_input = a_input.view(-1, E, E, 2*d)

            if self.q_attn:
                q_gate = F.relu(torch.matmul(query_vec, self.qattn_W1[i]))
                q_gate = torch.sigmoid(torch.matmul(q_gate, self.qattn_W2[i]))
                a_input = a_input * q_gate[:, None, None, :]
                score = self.act(torch.matmul(a_input, self.a_type[i]).squeeze(3))
            else:
                score = self.act(torch.matmul(a_input, self.a_type[i]).squeeze(3))

            scores += torch.where(adj == i+1, score, zero_vec.to(score.dtype))

        zero_vec = -1e30 * torch.ones_like(scores)
        scores = torch.where(adj > 0, scores, zero_vec.to(scores.dtype))

        # Ahead Alloc
        if node_mask is not None:
            h = h * node_mask

        coefs = F.softmax(scores, dim=2)  # N * E * E
        h = coefs.unsqueeze(3) * h.unsqueeze(2)  # N * E * E * d
        h = torch.sum(h, dim=1)
        return h


class AttentionLayer(nn.Module):
    def __init__(self, in_dim, hid_dim, n_head, q_attn, config):
        super(AttentionLayer, self).__init__()
        assert hid_dim % n_head == 0
        self.dropout = config.gnn_drop

        self.attn_funcs = nn.ModuleList()
        for i in range(n_head):
            self.attn_funcs.append(
                GATSelfAttention(in_dim=in_dim, out_dim=hid_dim // n_head, config=config, q_attn=q_attn, head_id=i))

        if in_dim != hid_dim:
            self.align_dim = nn.Linear(in_dim, hid_dim)
            nn.init.xavier_uniform_(self.align_dim.weight, gain=1.414)
        else:
            self.align_dim = lambda x: x

    def forward(self, input, adj, node_mask=None, query_vec=None):
        hidden_list = []
        for attn in self.attn_funcs:
            h = attn(input, adj, node_mask=node_mask, query_vec=query_vec)
            hidden_list.append(h)

        h = torch.cat(hidden_list, dim=-1)
        h = F.dropout(h, self.dropout, training=self.training)
        h = F.relu(h)
        return h


class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class OutputLayer(nn.Module):
    def __init__(self, hidden_dim, config, num_answer=1):
        super(OutputLayer, self).__init__()

        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.ReLU(),
            BertLayerNorm(hidden_dim*2, eps=1e-12),
            nn.Dropout(config.trans_drop),
            nn.Linear(hidden_dim*2, num_answer),
        )

    def forward(self, hidden_states):
        return self.output(hidden_states)

class GraphBlock(nn.Module):
    def __init__(self, q_attn, config):
        super(GraphBlock, self).__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim

        if self.config.q_update:
            self.gat_linear = nn.Linear(self.hidden_dim*2, self.hidden_dim)
        else:
            self.gat_linear = nn.Linear(self.config.input_dim, self.hidden_dim*2)

        if self.config.q_update:
            self.gat = AttentionLayer(self.hidden_dim, self.hidden_dim, config.num_gnn_heads, q_attn=q_attn, config=self.config)
            self.sent_mlp = OutputLayer(self.hidden_dim, config, num_answer=1)
            self.entity_mlp = OutputLayer(self.hidden_dim, config, num_answer=1)
        else:
            self.gat = AttentionLayer(self.hidden_dim*2, self.hidden_dim*2, config.num_gnn_heads, q_attn=q_attn, config=self.config)
            self.sent_mlp = OutputLayer(self.hidden_dim*2, config, num_answer=1)
            self.entity_mlp = OutputLayer(self.hidden_dim*2, config, num_answer=1)

    def forward(self, batch, input_state, query_vec):
        context_lens = batch['context_lens']
        context_mask = batch['context_mask']
        sent_mapping = batch['sent_mapping']
        sent_start_mapping = batch['sent_start_mapping']
        sent_end_mapping = batch['sent_end_mapping']
        para_mapping = batch['para_mapping']
        para_start_mapping = batch['para_start_mapping']
        para_end_mapping = batch['para_end_mapping']
        ent_mapping = batch['ent_mapping']
        ent_start_mapping = batch['ent_start_mapping']
        ent_end_mapping = batch['ent_end_mapping']

        def get_span_pooled_vec(state_input, mapping):
            mapping_state = state_input.unsqueeze(2) * mapping.unsqueeze(3)
            mapping_sum = mapping.sum(dim=1)
            mapping_sum = torch.where(mapping_sum == 0, torch.ones_like(mapping_sum), mapping_sum)
            mean_pooled = mapping_state.sum(dim=1) / mapping_sum.unsqueeze(-1)

            return mean_pooled

        para_start_output = torch.bmm(para_start_mapping, input_state[:, :, self.hidden_dim:])   # N x max_para x d
        para_end_output = torch.bmm(para_end_mapping, input_state[:, :, :self.hidden_dim])       # N x max_para x d
        para_state = torch.cat([para_start_output, para_end_output], dim=-1)  # N x max_para x 2d

        sent_start_output = torch.bmm(sent_start_mapping, input_state[:, :, self.hidden_dim:])   # N x max_sent x d
        sent_end_output = torch.bmm(sent_end_mapping, input_state[:, :, :self.hidden_dim])       # N x max_sent x d
        sent_state = torch.cat([sent_start_output, sent_end_output], dim=-1)  # N x max_sent x 2d

        ent_start_output = torch.bmm(ent_start_mapping, input_state[:, :, self.hidden_dim:])   # N x max_ent x d
        ent_end_output = torch.bmm(ent_end_mapping, input_state[:, :, :self.hidden_dim])       # N x max_ent x d
        ent_state = torch.cat([ent_start_output, ent_end_output], dim=-1)  # N x max_ent x 2d

        N, max_para_num, _ = para_state.size()
        _, max_sent_num, _ = sent_state.size()
        _, max_ent_num, _ = ent_state.size()

        if self.config.q_update:
            graph_state = self.gat_linear(torch.cat([para_state, sent_state, ent_state], dim=1)) # N * (max_para + max_sent + max_ent) * d
            graph_state = torch.cat([query_vec.unsqueeze(1), graph_state], dim=1)
        else:
            graph_state = self.gat_linear(query_vec)
            graph_state = torch.cat([graph_state.unsqueeze(1), para_state, sent_state, ent_state], dim=1)
        node_mask = torch.cat([torch.ones(N, 1).to(self.config.device), batch['para_mask'], batch['sent_mask'], batch['ent_mask']], dim=-1).unsqueeze(-1)

        graph_adj = batch['graphs']
        assert graph_adj.size(1) == node_mask.size(1)

        graph_state = self.gat(graph_state, graph_adj, node_mask=node_mask, query_vec=query_vec) # N x (1+max_para+max_sent) x d
        ent_state = graph_state[:, 1+max_para_num+max_sent_num:, :]

        gat_logit = self.sent_mlp(graph_state[:, :1+max_para_num+max_sent_num, :]) # N x max_sent x 1
        para_logit = gat_logit[:, 1:1+max_para_num, :].contiguous()
        sent_logit = gat_logit[:, 1+max_para_num:, :].contiguous()

        query_vec = graph_state[:, 0, :].squeeze(1)

        ent_logit = self.entity_mlp(ent_state).view(N, -1)
        ent_logit = ent_logit - 1e30 * (1 - batch['ans_cand_mask'])

        para_logits_aux = Variable(para_logit.data.new(para_logit.size(0), para_logit.size(1), 1).zero_())
        para_prediction = torch.cat([para_logits_aux, para_logit], dim=-1).contiguous()

        sent_logits_aux = Variable(sent_logit.data.new(sent_logit.size(0), sent_logit.size(1), 1).zero_())
        sent_prediction = torch.cat([sent_logits_aux, sent_logit], dim=-1).contiguous()

        return input_state, graph_state, node_mask, sent_state, query_vec, para_logit, para_prediction, \
            sent_logit, sent_prediction, ent_logit

class GatedAttention(nn.Module):
    def __init__(self, input_dim, memory_dim, hid_dim, dropout, gate_method='gate_att_up'):
        super(GatedAttention, self).__init__()
        self.gate_method = gate_method
        self.dropout = dropout
        self.input_linear_1 = nn.Linear(input_dim, hid_dim, bias=True)
        self.memory_linear_1 = nn.Linear(memory_dim, hid_dim, bias=True)

        self.input_linear_2 = nn.Linear(input_dim + memory_dim, hid_dim, bias=True)

        self.dot_scale = np.sqrt(input_dim)

    def forward(self, input, memory, mask):
        """
        :param input: context_encoding N * Ld * d
        :param memory: query_encoding N * Lm * d
        :param mask: query_mask N * Lm
        :return:
        """
        bsz, input_len, memory_len = input.size(0), input.size(1), memory.size(1)

        input_dot = F.relu(self.input_linear_1(input))  # N x Ld x d
        memory_dot = F.relu(self.memory_linear_1(memory))  # N x Lm x d

        # N * Ld * Lm
        att = torch.bmm(input_dot, memory_dot.permute(0, 2, 1).contiguous()) / self.dot_scale

        att = att - 1e30 * (1 - mask[:, None])
        weight_one = F.softmax(att, dim=-1)
        output_one = torch.bmm(weight_one, memory)

        if self.gate_method == 'no_gate':
            output = torch.cat( [input, output_one], dim=-1 )
            output = F.relu(self.input_linear_2(output))
        elif self.gate_method == 'gate_att_or':
            output = torch.cat( [input, input - output_one], dim=-1) 
            output = F.relu(self.input_linear_2(output))
        elif self.gate_method == 'gate_att_up':
            output = torch.cat([input, output_one], dim=-1 )
            gate_sg = torch.sigmoid(self.input_linear_2(output))
            gate_th = torch.tanh(self.input_linear_2(output))
            output = gate_sg * gate_th
        else:
            raise ValueError("Not support gate method: {}".format(self.gate_method))


        return output, memory


class BiAttention(nn.Module):
    def __init__(self, input_dim, memory_dim, hid_dim, dropout):
        super(BiAttention, self).__init__()
        self.dropout = dropout
        self.input_linear_1 = nn.Linear(input_dim, 1, bias=False)
        self.memory_linear_1 = nn.Linear(memory_dim, 1, bias=False)

        self.input_linear_2 = nn.Linear(input_dim, hid_dim, bias=True)
        self.memory_linear_2 = nn.Linear(memory_dim, hid_dim, bias=True)

        self.dot_scale = np.sqrt(input_dim)

    def forward(self, input, memory, mask):
        """
        :param input: context_encoding N * Ld * d
        :param memory: query_encoding N * Lm * d
        :param mask: query_mask N * Lm
        :return:
        """
        bsz, input_len, memory_len = input.size(0), input.size(1), memory.size(1)

        input = F.dropout(input, self.dropout, training=self.training)  # N x Ld x d
        memory = F.dropout(memory, self.dropout, training=self.training)  # N x Lm x d

        input_dot = self.input_linear_1(input)  # N x Ld x 1
        memory_dot = self.memory_linear_1(memory).view(bsz, 1, memory_len)  # N x 1 x Lm
        # N * Ld * Lm
        cross_dot = torch.bmm(input, memory.permute(0, 2, 1).contiguous()) / self.dot_scale
        # [f1, f2]^T [w1, w2] + <f1 * w3, f2>
        # (N * Ld * 1) + (N * 1 * Lm) + (N * Ld * Lm)
        att = input_dot + memory_dot + cross_dot  # N x Ld x Lm
        # N * Ld * Lm
        att = att - 1e30 * (1 - mask[:, None])

        input = self.input_linear_2(input)
        memory = self.memory_linear_2(memory)

        weight_one = F.softmax(att, dim=-1)
        output_one = torch.bmm(weight_one, memory)
        weight_two = F.softmax(att.max(dim=-1)[0], dim=-1).view(bsz, 1, input_len)
        output_two = torch.bmm(weight_two, input)

        return torch.cat([input, output_one, input*output_one, output_two*output_one], dim=-1), memory


class LSTMWrapper(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layer, concat=False, bidir=True, dropout=0.3, return_last=True):
        super(LSTMWrapper, self).__init__()
        self.rnns = nn.ModuleList()
        for i in range(n_layer):
            if i == 0:
                input_dim_ = input_dim
                output_dim_ = hidden_dim
            else:
                input_dim_ = hidden_dim if not bidir else hidden_dim * 2
                output_dim_ = hidden_dim
            self.rnns.append(nn.LSTM(input_dim_, output_dim_, 1, bidirectional=bidir, batch_first=True))
        self.dropout = dropout
        self.concat = concat
        self.n_layer = n_layer
        self.return_last = return_last

    def forward(self, input, input_lengths=None):
        # input_length must be in decreasing order
        bsz, slen = input.size(0), input.size(1)
        output = input
        outputs = []

        if input_lengths is not None:
            lens = input_lengths.data.cpu().numpy()

        for i in range(self.n_layer):
            output = F.dropout(output, p=self.dropout, training=self.training)

            if input_lengths is not None:
                output = rnn.pack_padded_sequence(output, lens, batch_first=True)

            output, _ = self.rnns[i](output)

            if input_lengths is not None:
                output, _ = rnn.pad_packed_sequence(output, batch_first=True)
                if output.size(1) < slen:  # used for parallel
                    padding = Variable(output.data.new(1, 1, 1).zero_())
                    output = torch.cat([output, padding.expand(output.size(0), slen-output.size(1), output.size(2))], dim=1)

            outputs.append(output)
        if self.concat:
            return torch.cat(outputs, dim=2)
        return outputs[-1]


class PredictionLayer(nn.Module):
    """
    Identical to baseline prediction layer
    """
    def __init__(self, config, q_dim):
        super(PredictionLayer, self).__init__()
        self.config = config
        input_dim = config.input_dim
        h_dim = config.hidden_dim

        self.hidden = h_dim

        self.start_linear = OutputLayer(input_dim, config, num_answer=1)
        self.end_linear = OutputLayer(input_dim, config, num_answer=1)
        self.type_linear = OutputLayer(input_dim, config, num_answer=4)

        self.cache_S = 0
        self.cache_mask = None

    def get_output_mask(self, outer):
        S = outer.size(1)
        if S <= self.cache_S:
            return Variable(self.cache_mask[:S, :S], requires_grad=False)
        self.cache_S = S
        np_mask = np.tril(np.triu(np.ones((S, S)), 0), 15)
        self.cache_mask = outer.data.new(S, S).copy_(torch.from_numpy(np_mask))
        return Variable(self.cache_mask, requires_grad=False)

    def forward(self, batch, context_input, sent_logits, packing_mask=None, return_yp=False):
        context_mask = batch['context_mask']
        context_lens = batch['context_lens']
        sent_mapping = batch['sent_mapping']

        sp_forward = torch.bmm(sent_mapping, sent_logits).contiguous()  # N x max_seq_len x 1

        start_prediction = self.start_linear(context_input).squeeze(2) - 1e30 * (1 - context_mask)  # N x L
        end_prediction = self.end_linear(context_input).squeeze(2) - 1e30 * (1 - context_mask)  # N x L
        type_prediction = self.type_linear(context_input[:, 0, :])

        if not return_yp:
            return start_prediction, end_prediction, type_prediction

        outer = start_prediction[:, :, None] + end_prediction[:, None]
        outer_mask = self.get_output_mask(outer)
        outer = outer - 1e30 * (1 - outer_mask[None].expand_as(outer))
        if packing_mask is not None:
            outer = outer - 1e30 * packing_mask[:, :, None]
        # yp1: start
        # yp2: end
        yp1 = outer.max(dim=2)[0].max(dim=1)[1]
        yp2 = outer.max(dim=1)[0].max(dim=1)[1]
        return start_prediction, end_prediction, type_prediction, yp1, yp2


def get_weights(size, gain=1.414):
    weights = nn.Parameter(torch.zeros(size=size))
    nn.init.xavier_uniform_(weights, gain=gain)
    return weights


def get_act(act):
    if act.startswith('lrelu'):
        return nn.LeakyReLU(float(act.split(':')[1]))
    elif act == 'relu':
        return nn.ReLU()
    else:
        raise NotImplementedError
    
def count_parameters(model, trainable_only=True, is_dict=False):
    """
    Count number of parameters in a model or state dictionary
    :param model:
    :param trainable_only:
    :param is_dict:
    :return:
    """
    if is_dict:
        return sum(np.prod(list(model[k].size())) for k in model)
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())
