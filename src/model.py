import torch
import transformers

import torch.nn as nn

def loss_fn(output, target, mask, num_labels):
    lfn = nn.CrossEntropyLoss()
    active_loss = mask.view(-1) == 1
    active_logits = output.view(-1, num_labels)
    active_labels = torch.where(
        active_loss,
        target.view(-1),
        torch.tensor(lfn.ignore_index).type_as(target)
    )
    loss = lfn(active_logits, active_labels)
    return loss

class Model(nn.Module):
  def __init__(self, num_tag):
    super(Model, self).__init__()
    self.num_tag = num_tag
    self.bert = transformers.BertModel.from_pretrained(
        "bert-base-uncased",
        return_dict = False
    )
    self.drop = nn.Dropout(0.3)
    self.linear = nn.Linear(768, self.num_tag)

  def forward(self, input_ids, token_type_ids, attention_mask, target):
    last_hidden_state, pooler_output = self.bert(
        input_ids = input_ids,
        token_type_ids = token_type_ids,
        attention_mask = attention_mask
    )

    last_hidden_state = self.drop(last_hidden_state)
    linear_output = self.linear(last_hidden_state)

    loss = loss_fn(linear_output, target, attention_mask, self.num_tag)

    return linear_output, loss