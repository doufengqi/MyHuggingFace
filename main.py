# 定义主函数 调用chi_mod.py中的chi_mod函数
import sys
import pandas as pd
from Mysql import mysql
from model.chi_mod import chi_mod
from tools.milvus_insert import milvus_insert
from model.all_MiniLM_L6_v2 import all_mini_l6v2
from tools.search_vector import search_vector


def main():
    print('请输入要查询的问题：')
    question = input()
    ###############################################################################
    # 选择进行向量化的模型
    print('请选择向量化模型：')
    print('1,Chinese_768')
    print('2,all_MiniLM_L6_v2')
    dims = 0
    model = input()
    if model == '1':
        print("开始执行 文本向量化")
        chi_mod()
        dims = 768
        print("文本向量化完成")
    elif model == '2':
        print("开始执行 文本向量化")
        dims = 384
        all_mini_l6v2()
        print("文本向量化完成")
    else:
        print("输入错误")
        sys.exit()
    ###############################################################################
    print("开始执行 向量插入Milvus")
    milvus_insert(dims)
    print("向量插入Milvus完成")
    ###############################################################################
    print("开始执行 文本数据插入Mysql")
    # 连接数据库 参数 库名 表名
    mysql('sentence_embeddings', 'sentence_embeddings')
    print("文本数据插入Mysql完成")
    ###############################################################################
    print("开始执行 Milvus检索")
    answer_list = search_vector(question)
    print("Milvus检索完成")

    answer_list = pd.array(answer_list)
    answer_list = answer_list.tolist()

    # 输出答案
    for i in range(len(answer_list)):
        print("第", i + 1, "个答案为：", answer_list[i])


if __name__ == '__main__':
    main()
