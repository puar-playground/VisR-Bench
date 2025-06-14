from tqdm import tqdm
import time
import json
import os
os.system('clear')
import clip
import torch
from PIL import Image
import argparse


def compute_clip_features(image_path, model, preprocess):
    """
    Computes the CLIP feature vector for a given image using OpenAI's CLIP.
    
    Parameters:
        image_path (str): Path to the image file.
        model_name (str): CLIP model name (default: "ViT-L/14").
        device (str): Device to run the model on ("cuda" or "cpu").
    
    Returns:
        torch.Tensor: CLIP feature vector of the image.
    """
    
    # Load and preprocess image
    image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)

    # Compute features
    with torch.no_grad():
        image_features = model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)  # Normalize


    return image_features.cpu()


def find_all_png_images(directory):
    """
    Traverses all subdirectories of a given directory and finds all PNG images.

    Parameters:
        directory (str): Path to the root directory.

    Returns:
        list: A list of full paths to all PNG images found.
    """
    png_images = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(".png"):  # Case insensitive check for PNG
                png_images.append(os.path.join(root, file))
    
    return png_images


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default=0, help="batch_index", type=int)
    parser.add_argument("--save_size", default=10000, help="save_size", type=int)
    args = parser.parse_args()

    # Load CLIP model
    device_index = int(args.index // 4)
    device = f'cuda:{device_index}'
    print(f'device: {device}')
    model, preprocess = clip.load("ViT-L/14", device=device)

    sub_folders = ['7', '4', 'e', '2', '1', 'a', '5', '9', 'b', '0', 'c', 'd', '8', '3', '6', 'f']
    f_index = sub_folders[args.index]

    batch_root_dir = f'CCpdf_Extract_{f_index}'
    print(f'batch_root_dir: {batch_root_dir}')

    png_images = find_all_png_images(f'/data/jian/CCPDF/{batch_root_dir}')
    l = len(png_images)
    print(f'{batch_root_dir}: len: {l}', )

    feature_tensor = torch.zeros([len(png_images), 768])
    for i, img_dir in tqdm(enumerate(png_images), ncols=100, total=l):
        try:
            image_features = compute_clip_features(img_dir, model, preprocess)
            feature_tensor[i, :] = image_features
        except:
            png_images[i] = 'failed'
            continue
        
        if (i+1) % args.save_size == 0:
            # Save features as a .pt file
            save_path = f'/data/jian/CCPDF_clip/{batch_root_dir}'
            
            torch.save(feature_tensor, save_path + '.pt')
            json.dump(png_images, open(save_path + '.json', 'w'), indent=2)
            print(f"Saved CLIP features to {save_path}")
        

    # Save features as a .pt file
    save_path = f'/data/jian/CCPDF_clip/{batch_root_dir}'
    
    torch.save(feature_tensor, save_path + '.pt')
    json.dump(png_images, open(save_path + '.json', 'w'), indent=2)
    print(f"Saved CLIP features to {save_path}")




