# This is a sentence-transformers model:
# It maps sentences & paragraphs to a 384 dimensional
# dense vector space and can be used for tasks like clustering or semantic search.

from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import csv


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# Sentences we want sentence embeddings for
# sentences = ['This is an example sentence', 'Each sentence is converted', '我是中国人']
# 从文件中读取句子
sentences = []
with open('sentences.txt', 'r', encoding='utf-8') as f:
    for line in f:
        sentences.append(line.strip())

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
#
# # 将句子向量插入到milvus中
#
# # Connect to Milvus server
# from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection
#
# connections.connect(host='127.0.0.1', port='19530')  # host为milvus的ip地址，port为milvus的端口号
#
# # Create collection
# collection_name = 'sentence_embeddings'
# fields = [
#     FieldSchema(name="int64", dtype=DataType.INT64, is_primary=True),
#     FieldSchema(name="float", dtype=DataType.FLOAT_VECTOR, dim=384)
# ]
# schema = CollectionSchema(fields=fields, description="sentence embeddings")
# collection = Collection(name=collection_name, schema=schema)
#
# # Insert data into Milvus
# with open('sentence_embeddings.csv', 'w', encoding='utf-8') as f:
#     writer = csv.writer(f)
#     for i in range(len(sentences)):
#         writer.writerow([i, sentence_embeddings[i].tolist()])
#         collection.insert([[int(i), sentence_embeddings[i].tolist()]])
#         print(i, sentences[i])
#
# # Close connection
# connections.close()

# 把向量和语句保存到csv 序号从0000000001开始
with open('sentence_embeddings.csv', 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    for i, sentence in enumerate(sentences):
        writer.writerow([i + 1, sentence, sentence_embeddings[i].tolist()])



# # 保存到txt 从0000000001开始
# with open('sentence_embeddings.txt', 'w', encoding='utf-8') as f:
#     for i, sentence in enumerate(sentences):
#         f.write(str(i + 1) + '\t' + str(sentence_embeddings[i].tolist()) + '\n')
