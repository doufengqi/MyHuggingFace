# This is a sentence-transformers model:
# It maps sentences & paragraphs to a 384 dimensional
# dense vector space and can be used for tasks like clustering or semantic search.
import csv
import xlrd
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


#
# # 从文件中读取句子
# sentences = []
# with open('sentences.txt', 'r', encoding='utf-8') as f:
#     for line in f:
#         sentences.append(line.strip())

# 从xls文件中读取句子
sentences = []
workbook = xlrd.open_workbook(r'../../data/sentences.xls')
# 读取第一行第一列begin
sheet = workbook.sheet_by_index(0)
begin = sheet.cell(0, 0).value
# 读取第二列存入sentences
for i in range(0, sheet.nrows):
    sentences.append(sheet.cell(i, 1).value)

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

# Tokenize sentences
encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

# Compute token embeddings
with torch.no_grad():
    model_output = model(**encoded_input)

# Perform pooling
sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

# Normalize embeddings
sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

print("Sentence embeddings:")
print(sentence_embeddings)

# 把向量和语句保存到csv 序号从0000000001开始
with open('sentence_embeddings.csv', 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    for i, sentence in enumerate(sentences):
        writer.writerow([i + 1, sentence, sentence_embeddings[i].tolist()])

# # 保存到txt
# with open('sentence_embeddings.txt', 'w', encoding='utf-8') as f:
#     for i, sentence in enumerate(sentences):
#         f.write(str(i + 1) + '\t' + str(sentence_embeddings[i].tolist()) + '\n')
