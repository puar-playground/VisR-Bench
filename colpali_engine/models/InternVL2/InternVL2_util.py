import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
import base64
import requests
from io import BytesIO
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
import argparse

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_file, input_size=448, max_num=20):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


# multi-image multi-round conversation, separate images (多图多轮对话，独立图像)
def ask(model, img_list, text_prompt):
    
    generation_config = dict(max_new_tokens=1024, do_sample=False)
    
    pixel_value_list = [load_image(img_dir, max_num=20).to(torch.bfloat16).cuda() for img_dir in img_list]
    pixel_values = torch.cat(pixel_value_list, dim=0)
    num_patches_list = [pixel_values.size(0) for pixel_values in pixel_value_list]

    prompt_img_prefix = '\n'.join([f'Image-{x}: <image>' for x in range(1, len(img_list)+1)])
    
    question = prompt_img_prefix + f'\n{text_prompt}'
    with torch.no_grad():
        response, history = model.chat(tokenizer, pixel_values, question, generation_config,
                                       num_patches_list=num_patches_list,
                                       history=None, return_history=True)
    print(f'text_prompt: {text_prompt}\nAssistant: {response}')
    return response

def ask_cat(model, img_list, text_prompt):
    
    # multi-image multi-round conversation, combined images (多图多轮对话，拼接图像)
    generation_config = dict(max_new_tokens=1024, do_sample=False)

    pixel_value_list = [load_image(img_dir, max_num=20).to(torch.bfloat16).cuda() for img_dir in img_list]
    pixel_values = torch.cat(pixel_value_list, dim=0)
    
    question = f'<image>\n{text_prompt}'
    response, history = model.chat(tokenizer, pixel_values, question, generation_config,
                                   history=None, return_history=True)
    print(f'text_prompt: {text_prompt}\nAssistant: {response}')
    return response


import json
import os
from PIL import Image
import numpy as np
from tqdm import tqdm

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data


def extract_number(filename):
    filename = filename.replace('-1024.jpg', '')
    x = filename.split('-')[-1]
    return int(x)


if __name__=="__main__":
    os.system('clear')
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', default='test', help='split', type=str)
    parser.add_argument('--size', default='8', help='model size in Billion', type=str)
    parser.add_argument('--max_num', default=20, help='max_num', type=int)
    parser.add_argument('--output_dir', default=None, help='output directory', type=str)
    args = parser.parse_args()
    
    # If you have an 80G A100 GPU, you can put the entire model on a single GPU.
    # Otherwise, you need to load a model using multiple GPUs, please refer to the `Multiple GPUs` section.
    path = f'OpenGVLab/InternVL2-{args.size}B' # 'OpenGVLab/InternVL-Chat-V1-5' # OpenGVLab/InternVL2-8B
    model = AutoModel.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map="auto"
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
    

    split = args.split
    img_path = f'/mnt/localssd/slidevqa_images/{split}'
    qa_path = f'/sensei-fs/users/jianc/VQA_dataset/SlideVQA/annotations/qa/{split}.jsonl'
    qa_data = load_jsonl(qa_path)
    
    cnt = len(qa_data)
    results = []
    correct = []
    pbar = tqdm(range(0, cnt), total=cnt, ncols=100)
    for p in pbar:
        qa = qa_data[p]
        deck_name = qa['deck_name']
        q_str = qa['question']
        evidence_pages = [int(x) for x in qa['evidence_pages']]
    
        if len(evidence_pages) != 1 and not os.path.isdir(os.path.join(img_path, deck_name)):
            continue
            
        img_dir_list = [x for x in os.listdir(os.path.join(img_path, deck_name)) if x.endswith('.jpg')]
        img_dir_list = sorted(img_dir_list, key=extract_number)
        img_dir_list = [os.path.join(img_path, deck_name, img) for img in img_dir_list]

        if len(img_dir_list) < 1:
            continue
    
        prompt = (f'Answer the question:\n{q_str}\nPlease make the answer as concise as possible')
    
        model_answer = ask_cat(model, img_dir_list, prompt)
        qa['model_answer'] = model_answer
        results.append(qa)
    
        json.dump(results, open(os.path.join(args.output_dir, f'qa_InternVL2_{args.size}B_{split}.json'), 'w'), indent=4)
