from text2vec import SentenceModel

sentences = []
with open('../data/sentences.txt', 'r', encoding='utf-8') as f:
    for line in f:
        sentences.append(line.strip())

model = SentenceModel('shibing624/text2vec-base-chinese')
embeddings = model.encode(sentences)
print(embeddings)
