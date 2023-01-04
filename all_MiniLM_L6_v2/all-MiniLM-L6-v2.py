from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F


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

# 保存到milvus
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection

# 连接milvus
connections.connect(host='127.0.0.1', port='19530')  # host为milvus的ip地址，port为milvus的端口号

# 创建collection 用于存储句子向量
collection_name = 'sentence_embedding'
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384)  # 句子向量 384维
]
schema = CollectionSchema(fields, description="sentence embedding")  # 创建collection 用于存储句子向量
collection = Collection(name=collection_name, schema=schema)

# 插入数据
collection.insert([sentence_embeddings.tolist()])
print("insert data into milvus")

# 从milvus中检索句子
from pymilvus import connections, Collection, utility

# 连接milvus
connections.connect(host='127.0.0.1', port='19530')  # host为milvus的ip地址，port为milvus的端口号

# 创建collection 用于存储句子向量
collection_name = 'sentence_embedding'
collection = Collection(name=collection_name)

# 检索句子
query = '到底该怎么定义'
query_embedding = model(**tokenizer(query, padding=True, truncation=True, return_tensors='pt'))[0][0]
query_embedding = F.normalize(query_embedding, p=2, dim=1)
query_embedding = query_embedding.tolist()
print("query embedding:")
print(query_embedding)
#
# # 检索
# # 向量检索
# top_k = 5
# search_params = {"metric_type": "L2", "params": {"nprobe": 10}}  # 表示 L2 距离，nprobe 表示搜索时的探测次数
# # search参数分别
# results = collection.search(query_embedding, "embedding", search_params, top_k)










#
#
# # 将id和sentence对应起来存入mysql
# import pymysql
#
# # 连接mysql
# db = pymysql.connect(host='127.0.0.1', port=3306, user='root', password='123456', db='sentence_embeddings',
#                      charset='utf8')
# cursor = db.cursor()
#
# # 创建表
# sql = 'create table sentence_embeddings2(id int primary key auto_increment, sentence varchar(255), embedding varchar(255))'
# cursor.execute(sql)
#
# # 将sentence_embeddings中的数据插入到mysql中
# sql = 'insert into sentence_embeddings(sentence, embedding) values(%s, %s)'
# for i in range(len(sentences)):
#     cursor.execute(sql, (sentences[i], str(sentence_embeddings[i][1])))
# db.commit()
# db.close()

#
# ###########################################################################################
# # 保存到文件
# with open('sentence_embeddings.txt', 'w', encoding='utf-8') as f:
#     for i in range(len(sentences)):
#         f.write(sentences[i] + '\t' + str(sentence_embeddings[i][1]) + '\n')
#
# # 保存到mysql
# import pymysql
#
# # 连接mysql
# db = pymysql.connect(host='127.0.0.1', port=3306, user='root', password='123456', db='sentence_embeddings',
#                      charset='utf8')
# cursor = db.cursor()
#
# # 创建表
# sql = 'create table sentence_embeddings (sentence varchar(255), embedding varchar(255))'
# cursor.execute(sql)
# db.commit()
#
# # 插入数据
# for i in range(len(sentences)):
#     sql = 'insert into sentence_embeddings (sentence, embedding) values (%s, %s)'
#     cursor.execute(sql, (sentences[i], str(sentence_embeddings[i][1])))
# db.commit()
#
# # 关闭连接
# cursor.close()
# db.close()
