from transformers import AutoTokenizer
from datasets import load_dataset
from itertools import chain
import numpy as np
import os

tokenizer_path = "/home/ps/llm/pretrain/tokenizer"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
print("成功加载tokenizer")
vocab_size = len(tokenizer)
print(f"词表大小为: {vocab_size}")
news_path = "/home/ps/llm/pretrain/dataset/news.txt"
raw_datasets = load_dataset(
    "text",  # 指定加载纯文本类型的数据
    data_files={"train": news_path}, # file_path 就是 "pretrain_hq.txt"
    split="train", # 指定加载哪个部分（如果数据文件有多个split定义）
    keep_in_memory=False
)

total_chars_in_raw_text = sum(len(text_line) for text_line in raw_datasets['text'])
print(f"原始文本总字符数: {total_chars_in_raw_text}")

def tokenizer_func(example):
    return tokenizer(example['text'], truncation=False, padding=False, add_special_tokens=False)


tokenized_datasets = raw_datasets.map(
    tokenizer_func,
    batched=True,
    remove_columns=["text"],
    desc=f"tokenizing",
    num_proc=32
)
all_token_ids = np.array(list(chain(*tokenized_datasets['input_ids'])), dtype=np.uint32)
print(len(all_token_ids))
print(f"压缩率 (tokens/lines):{len(all_token_ids)/len(raw_datasets['text']):.4f}")
if total_chars_in_raw_text > 0:
    print(f"压缩率 (tokens/chars):{len(all_token_ids)/total_chars_in_raw_text:.4f}")
all_token_ids.tofile(os.path.join(os.path.dirname(__file__), "news.bin"))


data = np.memmap(os.path.join(os.path.dirname(__file__), 'news.bin'), dtype=np.uint32, mode='r')
print(len(data))





