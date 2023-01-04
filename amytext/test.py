# from transformers import BertTokenizer, BertModel
# import torch
#
#
# # 平均池化 - 考虑注意力掩码以进行正确的平均
# def mean_pooling(model_output, attention_mask):
#     token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
#     input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
#     return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
#
#
# # Load model from HuggingFace Hub
# tokenizer = BertTokenizer.from_pretrained('shibing624/text2vec-base-chinese')
# model = BertModel.from_pretrained('shibing624/text2vec-base-chinese')
# sentences = ['如何更换花呗绑定银行卡', '花呗更改绑定银行卡']
# # Tokenize sentences
# encoded_input = tokenizer(sentences, padding=True, truncation=True,
#                           return_tensors='pt')  # 参数分别是输入，是否补全，是否截断，返回的数据类型 pt是pytorch的意思
#
# # 计算令牌嵌入
# with torch.no_grad():
#     model_output = model(**encoded_input)
# # Perform pooling. In this case, max pooling.
# sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
# print("Sentence embeddings:")
# print(sentence_embeddings)
