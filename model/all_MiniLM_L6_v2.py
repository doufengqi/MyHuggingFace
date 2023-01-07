import csv
import xlrd
import xlwt as xlwt
from text2vec import SentenceModel
from sentence_transformers import SentenceTransformer


def all_mini_l6v2():
    # 从xls文件中读取句子
    sentences = []
    workbook = xlrd.open_workbook(r'data/sentences.xls')
    # 读取第一行第一列begin
    sheet = workbook.sheet_by_index(0)
    begin = sheet.cell(0, 0).value
    # 读取第二列存入sentences
    for i in range(0, sheet.nrows):
        sentences.append(sheet.cell(i, 1).value)

    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = model.encode(sentences)
    print(embeddings)

    # 将向量写入csv文件 第一列为id 第二列为向量 id从begin开始
    # 一行一行写入
    with open('data/out/vector_out.csv', 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        for i in range(len(sentences)):
            writer.writerow([begin + i, sentences[i], embeddings[i].tolist()])
