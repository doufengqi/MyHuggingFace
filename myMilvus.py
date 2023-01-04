
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




