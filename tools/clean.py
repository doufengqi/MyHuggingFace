import pymysql
from pymilvus import connections, Collection, utility


def clean():
    # 连接milvus
    connections.connect(host='127.0.0.1', port='19530')
    if utility.has_collection('c_talk_test'):
        Collection('c_talk_test').drop()

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
