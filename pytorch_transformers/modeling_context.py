from __future__ import absolute_import, division, print_function, unicode_literals

import json
import logging
import math
import os
import sys
from io import open

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss


class ContextModel:
    def __init__(self, model, context_model):
        self.model = model
        self.context_model = context_model
        self.num_choices = 5

    def concat(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None):
        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        # flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1))

        outputs = self.model.roberta(flat_input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                               attention_mask=flat_attention_mask, head_mask=head_mask)
        context_outputs = self.context_model.roberta(flat_input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                                     attention_mask=flat_attention_mask, head_mask=head_mask)
        sequence_output = outputs[0]
        context_sequence_output= context_outputs[0]
        logits = self.model.classifier(sequence_output + context_sequence_output)

        reshaped_logits = logits.view(-1, self.num_choices)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
            return loss
        else:
            return reshaped_logits