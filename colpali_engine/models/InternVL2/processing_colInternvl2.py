from typing import ClassVar, List, Optional, Tuple, Union

import torch
from torch import nn
from PIL import Image

from transformers import BatchFeature
from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor

from PIL import Image
from typing import List

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from io import BytesIO
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, AutoModel

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


def load_image(input_image: Image.Image, input_size=448, max_num=20):
    
    input_image = input_image.convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(input_image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values



class ColInternProcessor(BaseVisualRetrieverProcessor):
    """
    Processor for ColPali.
    """
    
    visual_prompt_prefix: ClassVar[str] = "<|user|>\n<|image_1|>\nDescribe the image.<|end|>\n<|assistant|>\n"
    query_prefix: ClassVar[str] = "Query: "

    def __init__(self, model_name='OpenGVLab/InternVL2-4B', num_image_token=256, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.num_image_token: int = num_image_token
        
    @property
    def query_augmentation_token(self) -> str:
        """
        Return the query augmentation token.
        Query augmentation buffers are used as reasoning buffers during inference.
        """
        return self.tokenizer.pad_token
    

    def process_images(self, img: Image.Image):

        IMG_START_TOKEN='<img>'
        IMG_END_TOKEN='</img>'
        IMG_CONTEXT_TOKEN='<IMG_CONTEXT>'

        pixel_values = load_image(img).to(torch.bfloat16).to('cpu')
            
        n_patch = pixel_values.shape[0]
        pixel_values = torch.unsqueeze(pixel_values, dim=0)

        prompt = f'<image>\nDescribe the image.'
        image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * n_patch + IMG_END_TOKEN
        prompt = prompt.replace('<image>', image_tokens, 1)
        
        model_inputs = self.tokenizer(prompt, return_tensors='pt')
        input_ids = model_inputs['input_ids'].to('cpu')
    
        image_flags=torch.tensor([1] * n_patch, dtype=torch.long).to('cpu')
        image_flags = torch.unsqueeze(image_flags, dim=0) 

        meta = {'pixel_values': pixel_values, 'input_ids': input_ids, 'image_flags': image_flags}

        return meta #BatchFeature(data=meta)

    def process_queries(self, queries: List[str]):

        model_inputs = self.tokenizer(queries, return_tensors='pt', padding="longest",)
    
        return model_inputs #BatchFeature(data=model_inputs)

    def score(
        self,
        qs: List[torch.Tensor],
        ps: List[torch.Tensor],
        device: Optional[Union[str, torch.device]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute the MaxSim score (ColBERT-like) for the given multi-vector query and passage embeddings.
        """
        return self.score_multi_vector(qs, ps, device=device, **kwargs)

    def get_n_patches(
        self,
        image_size: Tuple[int, int],
        patch_size: int,
    ) -> Tuple[int, int]:
        n_patches_x = self.image_processor.size["width"] // patch_size
        n_patches_y = self.image_processor.size["height"] // patch_size

        return n_patches_x, n_patches_y

    def get_image_mask(self, batch_images: BatchFeature) -> torch.Tensor:
        return batch_images.input_ids == self.image_token_id
