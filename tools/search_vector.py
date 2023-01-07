import csv
import time
import numpy as np
import pandas as pd
import pymysql
from text2vec import SentenceModel
from pymilvus import connections, Collection, utility
from sklearn.metrics.pairwise import cosine_similarity


def search_vector(question):
    # 连接milvus
    connections.connect(host='localhost', port='19530')  # host为milvus的ip地址，port为milvus的端口号

    # 创建collection
    collection_name = 'c_talk_test'
    collection = Collection(name=collection_name)

    # 加载collection到内存
    collection.load()

    # 读取文本
    sentence = [question]

    # 转为向量
    model = SentenceModel('shibing624/text2vec-base-chinese')
    embeddings = model.encode(sentence)

    vector = embeddings[0].tolist()

    # 在milvus中查询相似度最高的向量并返回id 与数据库中的向量进行比较 从而得到相似度最高的文本
    top_k = 10
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    results = collection.search(
        data=[vector],
        anns_field="float",
        param=search_params,
        limit=top_k,
        # expr=None #expr指定查询条件 如id>100 and id<200
    )
    # print(results)  # 返回的结果为一个list 里面包含了一个dict 里面包含了id和distance

    collection.release()
    connections.disconnect(alias='default')

    # 连接数据库
    conn = pymysql.connect(
        host='localhost',
        port=3306,
        user='root',
        password='123456',
        db='sentence_embeddings',
        charset='utf8'
    )
    cursor = conn.cursor()

    # 读取results中的id存于id_list中
    id_list = []
    for i in range(len(results[0])):
        id_list.append(results[0][i].id)

    # 通过id_list中的id在mysql中id列中查询，将查询结果的answer列存于answer_list中
    answer_list = []
    for i in range(len(id_list)):
        cursor.execute("SELECT answer FROM sentence_embeddings WHERE id = %s", (id_list[i]))
        answer_list.append(cursor.fetchone()[0])
    # 关闭数据库连接
    cursor.close()
    conn.close()

    # print(answer_list)
    # 保存结果到csv文件中 命名为当前时间+answer_list
    df = pd.DataFrame(answer_list)
    df.to_csv('data/out/{}_answer.csv'.format(time.strftime("%Y%m%d%H%M%S", time.localtime())), index=False,
              header=False, encoding='utf-8')

    return answer_list
