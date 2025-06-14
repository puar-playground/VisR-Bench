import os
from PIL import Image
import torch
import requests
from transformers import CLIPProcessor, CLIPModel

device = "cuda"
torch_dtype = torch.float16

model = CLIPModel.from_pretrained(
    "openai/clip-vit-base-patch32",
    torch_dtype=torch_dtype,
)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
folders = []

for folder in folder_list:
    included = 0
    files = os.listdir(data_folder + folder)
    image_files = []
    for file in files:
        if '.png' in file:
            image_files.append(file)
    
    for index in range(len(image_files)):
        image = Image.open(os.path.join(data_folder, folder, image_files[index]))
        if image.size[0] < 256:
            continue
        inputs = processor(text=["Data Chart", "Scan Table", "Diagram", "Workflow", "Word Art", "Signature", "Infographics", "Photograph", "Equation", "Screenshots", "List", "Illustration", "Button", "Icon", "Logo", "Banner", "Background", "Map", "Drop cap", "Page Scans"], images=image, return_tensors="pt", padding=True)
        inputs.to(device)
        
        with torch.no_grad():
            with torch.autocast(device):
                outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
        probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
        # print(probs)
        if probs[0][0] > 0.5:
            included += 1
    if included:
        print(str(included) + ': ' + folder)
        folders.append(folder)