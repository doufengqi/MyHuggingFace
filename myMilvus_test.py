# import csv
# from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection
#
# # 连接milvus
# connections.connect(host='127.0.0.1', port='19530')  # host为milvus的ip地址，port为milvus的端口号
#
# # 创建collection
# collection_name = 'test_collection2'
# fields = [
#     FieldSchema(name="int64", dtype=DataType.INT64, is_primary=True, auto_id=False),
#     FieldSchema(name="float", dtype=DataType.FLOAT_VECTOR, dim=384),
#     # FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=1000)
# ]
# schema = CollectionSchema(fields=fields, description="test collection")
# collection = Collection(name=collection_name, schema=schema)
#
# # csv第一列为id，第二列为文本，第三列为向量
# with open('sentence_embeddings.csv', 'r', encoding='utf-8') as f:
#     reader = csv.reader(f)
#     for i, row in enumerate(reader):
#         # 强制转换为int64
#         row[0] = int(row[0])
#         # 强制转换为float_vector
#         row[2] = eval(row[2])
#         # 插入数据到milvus
#         collection.insert([[row[2]]])
#         print(i, row[1])
