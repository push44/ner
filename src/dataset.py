import config
import torch

class Dataset:
  def __init__(self, text_corpus, tag_corpus):
    self.text_corpus = text_corpus
    self.tag_corpus = tag_corpus

  def __len__(self):
    return len(self.text_corpus)

  def __getitem__(self, item):
    text = self.text_corpus[item]
    tags = self.tag_corpus[item]

    input_ids = []
    target_tags = []

    for ind, word in enumerate(text):
      inputs = config.TOKENIZER.encode(
          word,
          add_special_tokens = False
      )

      input_len = len(inputs)

      input_ids.extend(inputs)
      target_tags.extend([tags[ind]]*input_len)

    input_ids = input_ids[:config.MAX_LEN - 2]
    target_tags = target_tags[:config.MAX_LEN - 2]

    input_ids = [101] + input_ids + [102]
    target_tags = [0] + target_tags + [0]

    token_type_ids = [0]*len(input_ids)
    attention_mask = [1]*len(input_ids)

    padding_len = config.MAX_LEN - len(input_ids)

    input_ids = input_ids + ([0] * padding_len)
    token_type_ids = token_type_ids + ([0] * padding_len)
    attention_mask = attention_mask + ([0] * padding_len)
    target_tags = target_tags + ([0] * padding_len)

    #print(TOKENIZER.convert_ids_to_tokens(input_ids))

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "target": torch.tensor(target_tags, dtype=torch.long)
    }