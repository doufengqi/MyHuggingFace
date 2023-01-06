import csv
import xlrd
import xlwt as xlwt
from text2vec import SentenceModel


def chi_mod():
    # 从xls文件中读取句子
    sentences = []
    workbook = xlrd.open_workbook(r'../../data/sentences.xls')
    # 读取第一行第一列begin
    sheet = workbook.sheet_by_index(0)
    begin = sheet.cell(0, 0).value
    # 读取第二列存入sentences
    for i in range(0, sheet.nrows):
        sentences.append(sheet.cell(i, 1).value)

    model = SentenceModel('shibing624/text2vec-base-chinese')
    embeddings = model.encode(sentences)
    # print(embeddings)

    # 将向量写入csv文件 第一列为id 第二列为向量 id从begin开始
    # 一行一行写入
    with open('../../data/out/chiOut.csv', 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        for i in range(len(sentences)):
            writer.writerow([begin + i, sentences[i], embeddings[i].tolist()])

    #
    # # 创建xls并将向量写入 第一列为id 第二列为向量 id从begin开始
    # workbook = xlwt.Workbook()
    # sheet = workbook.add_sheet('sheet1')
    # for i in range(0, len(embeddings)):
    #     sheet.write(i, 0, begin + i)
    #     sheet.write(i, 1, sentences[i])
    #     sheet.write(i, 2, embeddings[i].tolist())
    # workbook.save('../../data/out/chiOut.xls')
