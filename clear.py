import pymysql
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, Milvus

# 连接milvus
connections.connect(host='127.0.0.1', port='19530')
# 查找c_talk_test 存在则删除
collections = connections.list_connections()

# if 'c_talk_test' in collections:
# Milvus().drop_collection('c_talk_test')
Collection('c_talk_test').drop()

collections.clear()
connections.disconnect(alias='default')

# 连接mysql
conn = pymysql.connect(
    host='127.0.0.1',
    port=3306,
    user='root',
    passwd='123456',
    charset='utf8'
)
# 查找sentence_embeddings数据库 存在则删除
cursor = conn.cursor()
cursor.execute('show databases')
databases = cursor.fetchall()
if ('sentence_embeddings',) in databases:
    cursor.execute('drop database sentence_embeddings')
