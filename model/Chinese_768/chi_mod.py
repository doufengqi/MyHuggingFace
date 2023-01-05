import csv
from text2vec import SentenceModel

sentences = []
with open('../data/sentences.txt', 'r', encoding='utf-8') as f:
    for line in f:
        sentences.append(line.strip())

model = SentenceModel('shibing624/text2vec-base-chinese')
embeddings = model.encode(sentences)
print(embeddings)


with open('../data/out/chiOut.csv', 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    for i, sentence in enumerate(sentences):
        writer.writerow([i + 1, sentence, embeddings[i].tolist()])