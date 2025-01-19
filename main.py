import argparse, os
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer
# ! direct tqdm is ok.
from tqdm import trange 

def pooling_max_token(model_output: torch.Tensor, attention_mask: torch.Tensor): 
    # 执行没有关系
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float() 
    # Set padding tokens to large negative value
    token_embeddings[input_mask_expanded == 0] = -1e9  
    max_vector = torch.max(token_embeddings, 1)[0]
    return max_vector


def pooling_mean_token(model_output, attention_mask):
    token_embeddings = model_output[0] # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def pooling_cls_token(model_output: torch.Tensor, attention_mask: torch.Tensor):
    token_embeddings = model_output[0]
    # Take first token by default
    cls_vector = token_embeddings[:, 0]
    return cls_vector


POOLING_MAP = {
    'cls': pooling_cls_token,
    'mean': pooling_mean_token,
    'max': pooling_max_token
}

#
# def cos_sim(a, b):
#     """
#     Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
#     :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
#     """
#     if not isinstance(a, torch.Tensor):
#         a = torch.tensor(a)
#
#     if not isinstance(b, torch.Tensor):
#         b = torch.tensor(b)
#
#     if a.dim() == 1:
#         a = a.unsqueeze(0)
#
#     if b.dim() == 1:
#         b = b.unsqueeze(0)
#
#     a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
#     b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
#     return torch.mm(a_norm, b_norm.transpose(0, 1))


def batch_to_device(dict_tensors, device='cpu'):
    result_tensors = {}
    for key, value in dict_tensors.items():
        result_tensors[key] = value.to(device)
    return result_tensors

# 
def encode(sentences, tokenizer, model, pooling='mean', batch_size=16):
    tokenizer_args = {'padding': True, 'truncation': True}
    all_embeddings = []
# query
    for start_index in trange(0, len(sentences), batch_size, desc="Batches"):
        sentences_batch = sentences[start_index:start_index + batch_size]
        # 用分词器进行分词，将结果显示直接为某个框架
        features = tokenizer(sentences_batch, return_token_type_ids=True, return_tensors='pt', **tokenizer_args)
        # 将数据移到GPU卡上
        features = batch_to_device(features, device='cpu')

        with torch.no_grad():
            # 模型进行编码，加入分词后的句子长度是s，当前组有32个句子，此时output的输出为32*s*768
            output = model(**features)
            # pooling层，假设当前组有32个句子，pooled_output大小为32*768
            # attention_mask 将句子统一长度，对于不同长度的句子进行补齐处理。
            # 结果输出到 map中
            pooled_output = POOLING_MAP[pooling](output, features['attention_mask'])
        # 保存当前batch的结果
        all_embeddings.extend(pooled_output)

    # 将分batch编码的结果进行合并 n * 768
    all_embeddings = torch.stack(all_embeddings)
    return all_embeddings

# 在命令行中显示最终结果
parser = argparse.ArgumentParser()
parser.add_argument('--data', default='D:/学习/科研助手/新建文件夹/answer.csv', type=str)
parser.add_argument('--model', default='./paraphrase-multilingual-mpnet-base-v2', type=str)
parser.add_argument('--pooling', default='mean', type=str, choices=['cls', 'mean', 'max'])
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--output_dir', default='results2', type=str)
args = parser.parse_args()
# 各个参数读取：
# 数据路径
data = args.data
# 模型路径
model = args.model
# Pooling的方式（模型把句子编码为多个token级别的embeddings，采取某种pooling的方式得到一个embedding来表示该句子）
pooling = args.pooling
# 编码时batch size大小
batch_size = args.batch_size
# 结果输出路径
output_dir = args.output_dir

# 1. load data
# 读取数据

df_sentence = pd.read_csv(data)
sentences = [sentence.get('sentence', sentence.get('picks')) for _, sentence in df_sentence.iterrows()]
n = len(sentences)
print(f"total sentences: {n}")

# 2. load model
# 读取模型，包括分词器tok和模型model
tok = AutoTokenizer.from_pretrained(model, model_max_length=512)
model = AutoModel.from_pretrained(model)
print("Model Loaded!")

# 3. encode
# 进行编码，目的是将每个句子转换成一个语义向量，例如有100个句子，模型的输出维度是768，则embddings的大小就是100*768，将每个句子转换成一个大小为768的向量
embeddings = encode(sentences, tok, model, pooling=pooling, batch_size=batch_size)
# we only consider the next sentence
# 我们只考虑相邻句子的相似度，假设有n个句子，next_sim输出大小为n-1的向量，其中第0个位置表示第0个句子和第1个句子的相似度。
next_sim = torch.cosine_similarity(embeddings[:-1], embeddings[1:])
result = pd.DataFrame(next_sim.cpu().detach().numpy())
print(result.head(10))
# save
#os.makedirs(output_dir, exist_ok=True)
# 保存于结果路径中相应的结果文件
"""
path = os.path.join(output_dir, 'result.xlsx')
writer = pd.ExcelWriter(path)
result.to_excel(writer, 'page_1', float_format='%.5f')
writer.save()
writer.close()
print(f"Saved to {path}!")
"""
