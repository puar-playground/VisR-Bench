from typing import ClassVar, List, Optional, Tuple, Union

import torch
from torch import nn

from PIL import Image

from transformers import BatchFeature
from colpali_engine.models.phi3v.processing_phi3_v import Phi3VProcessor

from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor

from PIL import Image, ImageOps
from typing import List

class ColPhiProcessor(BaseVisualRetrieverProcessor, Phi3VProcessor):
    """
    Processor for ColPali.
    """
    
    visual_prompt_prefix: ClassVar[str] = "<|user|>\n<|image_1|>\nDescribe the image.<|end|>\n<|assistant|>\n"
    query_prefix: ClassVar[str] = "Query: "
    

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer.padding_side = 'left'
        
    @property
    def query_augmentation_token(self) -> str:
        """
        Return the query augmentation token.
        Query augmentation buffers are used as reasoning buffers during inference.
        """
        return self.tokenizer.pad_token

    def process_images(
        self,
        images: List[Image.Image],
    ) -> BatchFeature:
        """
        Process images for ColPhi.
        """
        texts_doc = self.visual_prompt_prefix

        def pad_to_square(image: Image.Image) -> Image.Image:
            """Pad an image to make it square by adding padding to the smaller dimension."""
            width, height = image.size
            if width == height:
                return image
            else:
                max_dim = max(width, height)
                delta_w = max_dim - width
                delta_h = max_dim - height
                padding = (delta_w // 2, delta_h // 2, delta_w - delta_w // 2, delta_h - delta_h // 2)
                return ImageOps.expand(image, padding)
        
        images = [pad_to_square(image.convert("RGB")) for image in images]

        meta_list = [self(
            text=texts_doc,
            images=image,
            return_tensors="pt",
            padding="longest",
        ) for image in images]

        batch_doc = {key: torch.cat([meta[key] for meta in meta_list], dim=0) for key in meta_list[0]}
        
        return BatchFeature(data=batch_doc)

    def process_queries(
        self,
        queries: List[str],
        max_length: int = 50,
        suffix: Optional[str] = None,
    ) -> BatchFeature:
        """
        Process queries for ColPhi.
        """

        if suffix is None:
            suffix = self.query_augmentation_token * 10
        texts_query: List[str] = []

        for query in queries:
            query = self.tokenizer.bos_token + self.query_prefix + query
            query += suffix  # add suffix (pad tokens)
            query += "\n"

            texts_query.append(query)

        batch_query = self.tokenizer(
            texts_query,
            text_pair=None,
            return_token_type_ids=False,
            return_tensors="pt",
            padding="longest",
            max_length=max_length,
        )
        
        return BatchFeature(data=batch_query)

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
