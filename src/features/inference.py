""" This file is for inference of a NLP model """

from transformers import BigBirdPegasusForConditionalGeneration, AutoTokenizer
import requests
import PyPDF2

URL = "https://arxiv.org/pdf/1908.08593.pdf"
response = requests.get(URL)

tokenizer = AutoTokenizer.from_pretrained("google/bigbird-pegasus-large-pubmed")

# by default encoder-attention is `block_sparse` with num_random_blocks=3, block_size=64
model = BigBirdPegasusForConditionalGeneration.from_pretrained(
    "google/bigbird" "-pegasus-large-pubmed"
)

with open("1908.08593.pdf", "wb") as file:
    file.write(response.content)

with open("1908.08593.pdf", "rb") as file:
    reader = PyPDF2.PdfReader(file)
    page = reader.pages[0]
    text = page.extract_text()

inputs = tokenizer(text, return_tensors="pt")
prediction = model.generate(**inputs)
prediction = tokenizer.batch_decode(prediction)
print(prediction)
