import csv
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection

# 连接milvus
connections.connect(host='127.0.0.1', port='19530')  # host为milvus的ip地址，port为milvus的端口号

# 创建collection
collection_name = 'c_talk_test'
fields = [
    # 从1开始生成id
    FieldSchema(name="int64", dtype=DataType.INT64, is_primary=True, auto_id=False),  # id 为主键 且为int64类型 且为自增
    FieldSchema(name="float", dtype=DataType.FLOAT_VECTOR, dim=768),
]
schema = CollectionSchema(fields=fields, description="c_talk_test")
collection = Collection(name=collection_name, schema=schema)

with open('../data/out/chiOut.csv', 'r', encoding='utf-8') as f:
    # 读取csv第一行第一列和第一列最后一行存于begin和end中
    reader = csv.reader(f)
    my_vector = []
    # begin = next(reader)[0]
    for row in reader:
        if row[0] == '1':
            begin = row[0]
        my_vector.append(eval(row[2]))
    end = row[0]

    # 生成list 从begin到end
    my_id = [i for i in range(int(begin), int(end) + 1)]

    # 插入数据
    collection.insert([my_id, my_vector])

# 建立索引
# collection = Collection("c_talk_test")  # Get an existing collection.
# collection.create_index(
#     field_name="float",
#     index_params={
#         "metric_type": "L2",
#         "index_type": "IVF_FLAT",
#         "params": {"nlist": 1024}
#     }
# )
