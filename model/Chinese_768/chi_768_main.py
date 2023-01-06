# 定义主函数 调用chi_mod.py中的chi_mod函数
import sys
import os

sys.path.append(r'../../Mysql.py')

from Mysql import mysql
from chi_mod import chi_mod
from ChiMilvusInsert import milvus_insert


def main():
    print("开始执行 文本向量化")
    chi_mod()
    print("文本向量化完成")
    print("开始执行 向量插入Milvus")
    milvus_insert()
    print("向量插入Milvus完成")
    # 切换运行目录
    os.chdir(r'../..')
    print("开始执行 文本数据插入Mysql")
    mysql()
    print("文本数据插入Mysql完成")


if __name__ == '__main__':
    main()
