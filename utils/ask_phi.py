import time
import json
import os
os.system('clear')
from pdf2image import convert_from_path
import torch
from PIL import Image
from transformers import AutoModelForCausalLM 
from transformers import AutoProcessor
import torch

# Load Phi-3-V model
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "microsoft/Phi-3-vision-128k-instruct"
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).to(device)


def classify_page_with_phi3(image):
    """Classify a PDF page using Phi-3-V."""
    prompt = "Describe the contents of this image. Is it a table, a figure, a chart, a text-rich page, or mostly blank?"
    
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(**inputs)

    return processor.batch_decode(output, skip_special_tokens=True)[0]


def phi_classify_pdf_pages(images):
    """Classify each PDF page using Phi-3-V."""
    results = {}

    for idx, img in enumerate(images):
        page_number = idx + 1
        classification = classify_page_with_phi3(img)
        results[page_number] = classification

    return results

