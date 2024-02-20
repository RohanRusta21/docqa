# from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
# import PyPDF2

# model_name = "deepset/roberta-base-squad2"

# # Function to extract text from a PDF file
# def extract_text_from_pdf(pdf_path):
#     with open(pdf_path, 'rb') as file:
#         reader = PyPDF2.PdfReader(file)
#         text = ''
#         for page in reader.pages:
#             text += page.extract_text()
#     return text

# # Path to the PDF file
# pdf_path = 'magic.pdf'

# # Extract text from the PDF file
# context_text = extract_text_from_pdf(pdf_path)

# # Get predictions using pipeline
# nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)
# question = 'what is explained in Crop and cross marks paragraph?'  # Example question
# QA_input = {'question': question, 'context': context_text}
# res = nlp(QA_input)

# print(res)



import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

img_url = 'https://akm-img-a-in.tosshub.com/sites/visualstory/wp/2023/09/nun3.jpg?size=*:900' 
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

# conditional image captioning
# text = "a photography of"
# inputs = processor(raw_image, text, return_tensors="pt")

# out = model.generate(**inputs)
# print(processor.decode(out[0], skip_special_tokens=True))

# unconditional image captioning
inputs = processor(raw_image, return_tensors="pt")

out = model.generate(**inputs)
print(processor.decode(out[0], skip_special_tokens=True))
