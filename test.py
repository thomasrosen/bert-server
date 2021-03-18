from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained('bert-base-german-cased')
model = BertModel.from_pretrained('bert-base-german-cased')
text = 'Wie alt bist du?'
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)
print(output.pooler_output.detach().numpy())
