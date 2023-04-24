from transformers import BigBirdPegasusForConditionalGeneration, AutoTokenizer
import requests
import PyPDF2

url = 'https://www.example.com/example.pdf'
response = requests.get(url)

tokenizer = AutoTokenizer.from_pretrained("google/bigbird-pegasus-large-pubmed")

# by default encoder-attention is `block_sparse` with num_random_blocks=3, block_size=64
model = BigBirdPegasusForConditionalGeneration.from_pretrained("google/bigbird-pegasus-large-pubmed")

with open('example.pdf', 'wb') as file:
    file.write(response.content)

with open('example.pdf', 'rb') as file:
    reader = PyPDF2.PdfFileReader(file)
    page = reader.getPage(0)
    text = page.extractText()

    inputs = tokenizer(text, return_tensors='pt')
    prediction = model.generate(**inputs)
    prediction = tokenizer.batch_decode(prediction)
