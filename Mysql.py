import pymysql

conn = pymysql.connect(
    host='127.0.0.1',
    port=3306,
    user='root',
    passwd='123456',
    db='sentence_embeddings',
    charset='utf8'
)

cursor = conn.cursor()

# 创建表如果没有则创建 第一列为int类型，第二列为str类型
if not cursor.execute("show tables like 'sentence'"):
    cursor.execute("create table sentence(id int, sentence varchar(2000))")

# 将csv文件插入到数据库中 第一列为int类型，第二列为str类型 UTF-8
with open('sentence_embeddings.csv', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip().split(',')
        cursor.execute("insert into sentence values(%s, %s)", (line[0], line[1]))

conn.commit()
cursor.close()
conn.close()
