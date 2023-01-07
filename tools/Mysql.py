import xlrd
import pymysql


def mysql(database, table):
    # 如果库不存在则创建
    conn = pymysql.connect(
        host='127.0.0.1',
        port=3306,
        user='root',
        passwd='123456',
        charset='utf8'
    )

    # 检测mysql database库是否存在 如果不存在则创建 如果存在则跳过
    conn.cursor().execute("CREATE DATABASE IF NOT EXISTS %s DEFAULT CHARACTER SET utf8" % database)
    conn.commit()
    conn.close()

    # 连接数据库
    conn = pymysql.connect(
        host='127.0.0.1',
        port=3306,
        user='root',
        passwd='123456',
        charset='utf8',
        db=database
    )
    cursor = conn.cursor()

    # 如果没有则创建表sentence_embeddings 第一列为id 第二列为question 第三列为answer 如果有则不创建
    cursor.execute("CREATE TABLE IF NOT EXISTS sentence_embeddings "
                   "(id INT NOT NULL, question VARCHAR(255), answer VARCHAR(255), PRIMARY KEY (id))")

    # 插入数据
    # 打开xls文件并读取数据 插入到数据库中
    workbook = xlrd.open_workbook(r'data/sentences.xls')
    sheet = workbook.sheet_by_index(0)
    for i in range(0, sheet.nrows):
        id = sheet.cell(i, 0).value
        question = sheet.cell(i, 1).value
        answer = sheet.cell(i, 2).value
        cursor.execute("INSERT INTO sentence_embeddings VALUES (%s, %s, %s)", (id, question, answer))

    conn.commit()
    cursor.close()
    conn.close()
