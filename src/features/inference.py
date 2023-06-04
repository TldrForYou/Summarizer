""" This file is for inference of a NLP model """
import os
import argparse
from transformers import BigBirdPegasusForConditionalGeneration, AutoTokenizer
import requests
import PyPDF2
import mlflow
from mlflow.models.signature import infer_signature

os.environ["AWS_ACCESS_KEY_ID"] = ""  # put your credentials here
os.environ["AWS_SECRET_ACCESS_KEY"] = ""  # put your credentials here
os.environ["MLFLOW_S3_ENDPOINT_URL"] = f"http://89.19.208.000:9000"  # put server's ip

mlflow.set_tracking_uri("http://89.19.208.000:5000")
mlflow.set_experiment("test1")
mlflow.transformers.autolog()


def extract_text(input_url_pdf: str, output_path_pdf: str):
    """"Function to extract text from chosen pdf"""

    # URL = "https://arxiv.org/pdf/1908.08593.pdf"
    response = requests.get(input_url_pdf)
    with open(output_path_pdf, "wb") as file:
        file.write(response.content)

    with open(output_path_pdf, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        page = reader.pages[0]
        text = page.extract_text()

    return text

# by default encoder-attention is `block_sparse` with num_random_blocks=3,
# block_size=64


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='PDF Text Extraction and Model Inference')
    parser.add_argument('--input_url_pdf', required=True,
                        help='Input URL of the PDF file to extract text from')
    parser.add_argument('--output_path_pdf', help='Output path for the extracted text')
    parser.add_argument('--input_url_tokenizer', required=True,
                        help='Input URL of the tokenizer to use')
    parser.add_argument('--input_url_model', required=True,
                        help='Input URL of the model to use')
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.input_url_tokenizer)

    extr_text = extract_text(args.input_url_pdf, args.output_path_pdf)
    inputs = tokenizer(extr_text, return_tensors="pt")

    model = BigBirdPegasusForConditionalGeneration.from_pretrained(args.input_url_model)

    prediction = model.generate(**inputs)
    prediction = tokenizer.batch_decode(prediction)
    signature = infer_signature(extr_text, prediction)

    with mlflow.start_run():
        components = {
            "model": model,
            "tokenizer": tokenizer,
        }
        model_info = mlflow.transformers.log_model(
            transformers_model=components,
            artifact_path="summarizer",
            framework="pt",
            signature=signature,
            inference_config={"max_length": 256}
        )

    autolog_run = mlflow.last_active_run()
    mlflow.end_run()
    print(prediction)
