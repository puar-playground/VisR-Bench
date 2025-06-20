from PIL import Image
import os
import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from typing import List, Tuple, Union
os.system('clear')
from abc import ABC, abstractmethod
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# for clip model imports
import clip
# for siglip, NV-embed model imports
from utils.nvembed.modeling_nvembed import NVEmbedModel
from transformers import AutoProcessor, AutoModel, AutoTokenizer
# for BM25 imports
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
# SBERT
from sentence_transformers import SentenceTransformer
# GEM
from .GEM.gme_inference import GmeQwen2VL
# VLM2vec
from .VLM2Vec.src.model import MMEBModel
from .VLM2Vec.src.arguments import ModelArguments
from .VLM2Vec.src.utils import load_processor
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

class BaseRetriever(ABC):
    """
    A base class for retriever models that process both text and images 
    and compute similarities for retrieval tasks.
    """

    @abstractmethod
    def retrieve(self, query, image_list, top_k=5):
        """
        Retrieves the most relevant images from a list based on the given query.

        Args:
            query (str): The input query text.
            image_list (list of str): List of image file paths.
            top_k (int): Number of top relevant images to return (default: 5).

        Returns:
            list of tuples: Sorted list of (image_path, similarity_score), descending order.
        """
        pass

    @abstractmethod
    def compute_similarity(self, item1, item2):
        """
        Computes similarity between two items (text or image).
        
        Args:
            item1: First item (can be text or image).
            item2: Second item (can be text or image).
        
        Returns:
            A similarity score (float).
        """
        pass

    @abstractmethod
    def process_text(self, text):
        """
        Processes text input (e.g., tokenization, embedding).
        
        Args:
            text (str): List of input text strings to process.
        
        Returns:
            Processed text representation.
        """
        pass

    @abstractmethod
    def process_image(self, image_list):
        """
        Processes image input (e.g., feature extraction, encoding).
        
        Args:
            image_list: List of input image directories to process.
        
        Returns:
            Processed image representation.
        """
        pass

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
from colpali_engine.models import ColPali, ColPaliProcessor

class ColPaliRetriever(BaseRetriever):
    """Retriever class using ColPali for multimodal retrieval."""

    def __init__(self, model_name="vidore/colpali-v1.2", device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initializes the ColPali model.

        Args:
            model_name (str): The model identifier.
            device (str): Device to run the model on ('cuda' or 'cpu').
        """
        os.system('pip install transformers==4.47.1')

        self.multimodel = True
        self.device = device

        self.model = ColPali.from_pretrained(
            model_name,
                torch_dtype=torch.bfloat16,
                device_map="cuda:0",  # or "mps" if on Apple Silicon
            ).eval()
        self.processor = ColPaliProcessor.from_pretrained(model_name)

    def process_text(self, query_list: str):
        """Processes text into embeddings using ColPali."""
        
        # batch_images = process_images_in_batches(processor, img_dir_list, model)
        batch_queries = self.processor.process_queries(query_list).to(self.model.device)
        with torch.no_grad():
            querry_embeddings = self.model(**batch_queries)

        return querry_embeddings

    def process_image(self, image_dir_list: List[str]):
        """Processes images into embeddings using ColPali."""
        def process_images_in_batches(processor, img_dir_list, model, batch_size=2):
            all_embeddings = []
            
            # Split img_dir_list into batches
            for i in range(0, len(img_dir_list), batch_size):
                batch_img_dirs = img_dir_list[i:i + batch_size]
                image_list = [Image.open(img_dir) for img_dir in batch_img_dirs]
        
                # Process the batch of images
                batch_features = processor.process_images(image_list)
                
                # Extract the tensor from the BatchFeature object
                batch_images = {k: v.to(model.device) for k, v in batch_features.items()}
        
                # Assuming the model expects a specific input (e.g., 'pixel_values')
                embeddings = model(**batch_images)
                
                # Move embeddings to CPU and append to the list
                embeddings = embeddings.to("cpu")
                all_embeddings.append(embeddings)

            # Concatenate all processed batches into a single tensor
            all_embeddings = torch.cat(all_embeddings, dim=0)
            return all_embeddings
        
        # Forward pass
        with torch.no_grad():
            # image_embeddings = model(**batch_images)
            image_embeddings = process_images_in_batches(self.processor, image_dir_list, self.model)
                    
        return image_embeddings

    def compute_similarity(self, text_embeddings, image_embeddings):
        """ Computes cosine similarity between text and image embeddings. """
        scores = self.processor.score_multi_vector(text_embeddings, image_embeddings)
        return scores

    def retrieve(self, query_list: str, image_list: List[str]):

        text_embeddings = self.process_text(query_list)
        image_embeddings = self.process_image(image_list)

        similarity_score = self.compute_similarity(text_embeddings, image_embeddings)
        values, top_indices = torch.tensor(similarity_score).sort(descending=True)

        return values, top_indices


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
from colpali_engine.models import ColPhi, ColPhiProcessor

class ColPhiRetriever(BaseRetriever):
    """Retriever class using ColPhi for multimodal retrieval."""

    def __init__(self, model_name="puar-playground/Col-Phi-3-V", device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initializes the ColPhi model.

        Args:
            model_name (str): The model identifier.
            device (str): Device to run the model on ('cuda' or 'cpu').
        """
        os.system('pip install transformers==4.47.1')
        self.multimodel = True
        self.device = device

        self.model = ColPhi.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
        ).eval()
        self.processor = ColPhiProcessor.from_pretrained(model_name)

    @staticmethod
    def pad_and_cat_tensors(tensor_list):
        # Find the maximum length of the second dimension (x_i) across all tensors
        max_x = max(tensor.size(1) for tensor in tensor_list)
        
        # Pad tensors to have the same size in the second dimension
        padded_tensors = []
        for tensor in tensor_list:
            padding_size = max_x - tensor.size(1)
            # Pad with zeros on the right in the second dimension
            padded_tensor = torch.nn.functional.pad(tensor, (0, 0, 0, padding_size))
            padded_tensors.append(padded_tensor)
        
        # Concatenate the padded tensors along the first dimension
        result_tensor = torch.cat(padded_tensors, dim=0)
        
        return result_tensor
    
    def process_text(self, query_list: List[str], batch_size: int = 2):
        """
        Processes a list of text queries into embeddings using ColPhi in batches.

        Args:
            query_list (List[str]): List of query texts.
            batch_size (int): Number of queries processed per batch.

        Returns:
            torch.Tensor: Concatenated embeddings for all queries.
        """
        all_embeddings = []

        for i in range(0, len(query_list), batch_size):
            batch_queries = query_list[i : i + batch_size]

            # Convert queries to model-compatible format
            batch_inputs = self.processor.process_queries(batch_queries).to(self.model.device)

            with torch.no_grad():
                batch_embeddings = self.model(**batch_inputs)

            all_embeddings.append(batch_embeddings.to("cpu"))
        
        # Concatenate all processed batches into a single tensor
        all_embeddings = self.pad_and_cat_tensors(all_embeddings)

        # Concatenate all batch outputs into a single tensor
        return all_embeddings

    def process_image(self, image_dir_list: List[str]):
        """Processes images into embeddings using ColPhi."""
        def process_images_in_batches(processor, img_dir_list, model, batch_size=1):
            all_embeddings = []
            
            # Split img_dir_list into batches
            for i in range(0, len(img_dir_list), batch_size):
                batch_img_dirs = img_dir_list[i:i + batch_size]
                image_list = [Image.open(img_dir) for img_dir in batch_img_dirs]
        
                # Process the batch of images
                batch_features = processor.process_images(image_list)
                
                # Extract the tensor from the BatchFeature object
                batch_images = {k: v.to(model.device) for k, v in batch_features.items()}
        
                # Assuming the model expects a specific input (e.g., 'pixel_values')
                embeddings = model(**batch_images)
                
                # Move embeddings to CPU and append to the list
                embeddings = embeddings.to("cpu")
                all_embeddings.append(embeddings)

            # Concatenate all processed batches into a single tensor
            all_embeddings = torch.cat(all_embeddings, dim=0)
            return all_embeddings
        
        # Forward pass
        with torch.no_grad():
            # image_embeddings = model(**batch_images)
            image_embeddings = process_images_in_batches(self.processor, image_dir_list, self.model)
                    
        return image_embeddings

    def compute_similarity(self, text_embeddings, image_embeddings):
        """ Computes cosine similarity between text and image embeddings. """
        scores = self.processor.score_multi_vector(text_embeddings, image_embeddings)
        return scores

    def retrieve(self, query_list: str, image_list: List[str]):
        
        with torch.no_grad():
            text_embeddings = self.process_text(query_list)
            image_embeddings = self.process_image(image_list)

        similarity_score = self.compute_similarity(text_embeddings, image_embeddings)
        values, top_indices = torch.tensor(similarity_score).sort(descending=True)

        return values, top_indices

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
from colpali_engine.models import ColInternvl2_4b, ColInternProcessor

class ColInternVL2Retriever(BaseRetriever):
    """Retriever class using ColInternVL2 for multimodal retrieval."""

    def __init__(self, model_name="puar-playground/Col-InternVL2-4B", device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initializes the ColInternVL2 model.

        Args:
            model_name (str): The model identifier.
            device (str): Device to run the model on ('cuda' or 'cpu').
        """
        os.system('pip install transformers==4.47.1')
        self.multimodel = True
        self.device = device

        self.model = ColInternvl2_4b.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device).eval()

        self.processor = ColInternProcessor('OpenGVLab/InternVL2-4B')

    def process_text(self, query_list: List[str], batch_size: int = 4):
        """
        Processes a list of text queries into embeddings using ColPhi in batches.

        Args:
            query_list (List[str]): List of query texts.
            batch_size (int): Number of queries processed per batch.

        Returns:
            torch.Tensor: Concatenated embeddings for all queries.
        """
        all_embeddings = []

        for i in range(0, len(query_list), batch_size):
            batch_queries = query_list[i : i + batch_size]

            # Convert queries to model-compatible format
            batch_inputs = self.processor.process_queries(batch_queries).to(self.model.device)

            with torch.no_grad():
                batch_embeddings = self.model(**batch_inputs)

            all_embeddings.append(batch_embeddings.to("cpu"))
        
        # Concatenate all batch outputs into a single tensor
        all_embeddings = self.pad_and_cat_tensors(all_embeddings)

        return all_embeddings
    
    @staticmethod
    def pad_and_cat_tensors(tensor_list):
        # Find the maximum length of the second dimension (x_i) across all tensors
        max_x = max(tensor.size(1) for tensor in tensor_list)
        
        # Pad tensors to have the same size in the second dimension
        padded_tensors = []
        for tensor in tensor_list:
            padding_size = max_x - tensor.size(1)
            # Pad with zeros on the right in the second dimension
            padded_tensor = torch.nn.functional.pad(tensor, (0, 0, 0, padding_size))
            padded_tensors.append(padded_tensor)
        
        # Concatenate the padded tensors along the first dimension
        result_tensor = torch.cat(padded_tensors, dim=0)
        
        return result_tensor

    def process_image(self, image_dir_list: List[str]):
        """Processes images into embeddings using ColInternVL2."""
        def process_images_in_batches(processor, img_dir_list, model, batch_size=2):
            all_embeddings = []
            
            # Split img_dir_list into batches
            for img_dir in img_dir_list:

                img = Image.open(img_dir)
        
                # Process the batch of images
                batch_features = processor.process_images(img)
                
                # Extract the tensor from the BatchFeature object
                batch_images = {k: v.to(model.device) for k, v in batch_features.items()}
        
                # Assuming the model expects a specific input (e.g., 'pixel_values')
                embeddings = model(**batch_images)
                
                # Move embeddings to CPU and append to the list
                embeddings = embeddings.to("cpu")
                all_embeddings.append(embeddings)

            # Concatenate all processed batches into a single tensor
            all_embeddings = self.pad_and_cat_tensors(all_embeddings)
            return all_embeddings
        
        # Forward pass
        with torch.no_grad():
            # image_embeddings = model(**batch_images)
            image_embeddings = process_images_in_batches(self.processor, image_dir_list, self.model)
                    
        return image_embeddings

    def compute_similarity(self, text_embeddings, image_embeddings):
        """ Computes cosine similarity between text and image embeddings. """
        scores = self.processor.score_multi_vector(text_embeddings, image_embeddings)
        return scores

    def retrieve(self, query_list: str, image_list: List[str]):

        text_embeddings = self.process_text(query_list)
        image_embeddings = self.process_image(image_list)

        similarity_score = self.compute_similarity(text_embeddings, image_embeddings)
        values, top_indices = torch.tensor(similarity_score).sort(descending=True)

        return values, top_indices

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
class VisRAGRetriever(BaseRetriever):
    """Retriever class using OpenBMB's VisRAG for multimodal retrieval."""

    def __init__(self, model_name="openbmb/VisRAG-Ret", device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initializes the VisRAG model and tokenizer.

        Args:
            model_name (str): The model identifier.
            device (str): Device to run the model on ('cuda' or 'cpu').
        """
        self.multimodel = True
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, torch_dtype=torch.bfloat16, trust_remote_code=True).to(self.device)
        self.model.eval()

    @staticmethod
    def weighted_mean_pooling(hidden, attention_mask):
        """
        Computes weighted mean pooling over the hidden states.

        Args:
            hidden (torch.Tensor): Hidden state tensor.
            attention_mask (torch.Tensor): Attention mask tensor.

        Returns:
            torch.Tensor: Pooled and normalized embeddings.
        """
        attention_mask_ = attention_mask * attention_mask.cumsum(dim=1)
        s = torch.sum(hidden * attention_mask_.unsqueeze(-1).float(), dim=1)
        d = attention_mask_.sum(dim=1, keepdim=True).float()
        reps = s / d
        return F.normalize(reps, p=2, dim=1).detach()

    def process_text(self, text_list: List[str]):
        """
        Processes text input (e.g., tokenization, embedding).
        
        Args:
            text (str): List of input text strings to process.
        
        Returns:
            Processed text representation.
        """
        input_data = {
                "text": text_list,
                "image": [None] * len(text_list),
                "tokenizer": self.tokenizer
            }
        # Forward pass through model
        outputs = self.model(**input_data)
        attention_mask = outputs.attention_mask
        hidden = outputs.last_hidden_state

        return self.weighted_mean_pooling(hidden, attention_mask).cpu()

    def process_image(self, image_dir_list):
        """
        Processes image input (e.g., feature extraction, encoding).
        
        Args:
            image_list: List of input image directories to process.
        
        Returns:
            Processed image representation.
        """
        image_list = [Image.open(img_dir) for img_dir in image_dir_list]
        input_data = {
                "text": [''] * len(image_list),
                "image": image_list,
                "tokenizer": self.tokenizer
            }
        # Forward pass through model
        outputs = self.model(**input_data)
        attention_mask = outputs.attention_mask
        hidden = outputs.last_hidden_state

        return self.weighted_mean_pooling(hidden, attention_mask).cpu()
    
    def compute_similarity(self, text_features, image_features):
        """ Computes cosine similarity between text and image embeddings. """
        return (text_features @ image_features.T)
    
    def retrieve(self, query_list: List[str], image_list: List[Image.Image], batch_size: int=1) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes similarity scores between text queries and images.

        Args:
            query_list (list of str): List of query texts.
            image_list (list of Image.Image): List of image objects.

        Returns:
            values (torch.Tensor): Sorted similarity scores.
            indices (torch.Tensor): Sorted indices of most relevant images for each query.
        """
        if not query_list or not image_list:
            raise ValueError("Both query_list and image_list must be non-empty.")

        # Prepend retrieval instruction to queries
        def batch_encode_texts(texts):
            """Encodes texts in mini-batches to avoid memory overflow."""
            all_embeddings = []
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]
                with torch.no_grad():
                    embeddings = self.process_text(batch_texts)  # Shape: (batch_size, embedding_dim)
                    all_embeddings.append(embeddings)
                    
            return torch.cat(all_embeddings, dim=0)  # Shape: (num_texts, embedding_dim)

        def batch_encode_images(images):
            """Encodes images in mini-batches to avoid memory overflow."""
            all_embeddings = []
            for i in range(0, len(images), batch_size):
                batch_images = images[i : i + batch_size]
                with torch.no_grad():
                    embeddings = self.process_image(batch_images)  # Shape: (batch_size, embedding_dim)
                    all_embeddings.append(embeddings)
            return torch.cat(all_embeddings, dim=0)  # Shape: (num_images, embedding_dim)

        
        # Encode queries and images
        with torch.no_grad():
            # Encode queries and images in batches
            query_embeddings = batch_encode_texts(query_list)  # Shape: (num_queries, embedding_dim)
            image_embeddings = batch_encode_images(image_list)  # Shape: (num_images, embedding_dim)
            
        # Compute cosine similarity
        similarity_score = self.compute_similarity(query_embeddings, image_embeddings)

        # Sort results for each query
        values, indices = torch.sort(similarity_score, dim=1, descending=True)

        return values, indices

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
class VLM2VecRetriever(BaseRetriever):
    """Retriever class using TIGER-Lab/VLM2Vec-Full for multimodal retrieval."""

    def __init__(self, model_name="TIGER-Lab/VLM2Vec-Full", device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initializes the VLM2Vec model and processor.

        Args:
            model_name (str): The model identifier.
            device (str): Device to run the model on ('cuda' or 'cpu').
        """
        self.multimodel = True
        self.device = device
        self.model_args = ModelArguments(
            model_name=model_name,
            pooling="last",
            normalize=True,
            model_backbone="phi3_v",
            num_crops=16
        )

        self.processor = load_processor(self.model_args)
        self.model = MMEBModel.load(self.model_args).to(self.device, dtype=torch.bfloat16)
        self.model.eval()

    def process_text(self, text):
        """Processes text into embeddings using VLM2Vec."""
        inputs = self.processor(text)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        return self.model(qry=inputs)["qry_reps"]

    def process_image(self, image_dir_list):
        """Processes an image into embeddings using VLM2Vec."""
        image_list = [Image.open(img_dir) for img_dir in image_dir_list]
        inputs = self.processor('<|image_1|> Represent the given image.', image_list)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        return self.model(tgt=inputs)["tgt_reps"]
        

    def compute_similarity(self, item1, item2) -> float:
        """
        Computes cosine similarity between two embeddings.

        Args:
            item1 (torch.Tensor): Embedding of the first item.
            item2 (torch.Tensor): Embedding of the second item.

        Returns:
            float: Similarity score.
        """
        return self.model.compute_similarity(item1, item2).item()

    def retrieve(self, query_list: List[str], image_list: List[str], batch_size: int = 1):
        """
        Retrieves the most relevant candidates (images) based on the query list.

        Args:
            query_list (list of str): The list of query texts.
            image_list (list of str): The list of image file paths or URLs.
            batch_size (int): Batch size for encoding queries and images.

        Returns:
            values (torch.Tensor): Sorted similarity scores for each query.
            indices (torch.Tensor): Sorted indices of most relevant images for each query.
        """
        if not query_list or not image_list:
            raise ValueError("Both query_list and image_list must be non-empty.")

        def batch_encode_texts(texts):
            """Encodes texts in mini-batches to avoid memory overflow."""
            all_embeddings = []
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]
                with torch.no_grad():
                    embeddings = self.process_text(batch_texts)  # Shape: (batch_size, embedding_dim)
                    all_embeddings.append(embeddings)
                    
            return torch.cat(all_embeddings, dim=0)  # Shape: (num_texts, embedding_dim)

        def batch_encode_images(images):
            """Encodes images in mini-batches to avoid memory overflow."""
            all_embeddings = []
            for i in range(0, len(images), batch_size):
                batch_images = images[i : i + batch_size]
                with torch.no_grad():
                    embeddings = self.process_image(batch_images)  # Shape: (batch_size, embedding_dim)
                    all_embeddings.append(embeddings)
            return torch.cat(all_embeddings, dim=0)  # Shape: (num_images, embedding_dim)

        # Encode queries and images in batches
        query_embeddings = batch_encode_texts(query_list)  # Shape: (num_queries, embedding_dim)
        image_embeddings = batch_encode_images(image_list)  # Shape: (num_images, embedding_dim)

        # Compute similarity scores (cosine similarity)
        similarity_scores = torch.matmul(query_embeddings, image_embeddings.T)  # Shape: (num_queries, num_images)

        # Sort results for each query
        values, indices = torch.sort(similarity_scores, descending=True)

        return values, indices  # Sorted similarity scores and indices

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

class GmeRetriever(BaseRetriever):
    """Retriever class using Alibaba's GME-Qwen2-VL model for multimodal retrieval."""

    def __init__(self, model_name="Alibaba-NLP/gme-Qwen2-VL-2B-Instruct", device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initializes the GME-Qwen2-VL model.

        Args:
            model_name (str): Model identifier from Hugging Face.
            device (str): Device to run the model on ('cuda' or 'cpu').
        """
        os.system('pip install --upgrade transformers')
        self.multimodel = True
        self.device = device
        self.model = GmeQwen2VL(model_name)
    
    def process_text(self, query_list: str):
        """Processes text into embeddings using GME-Qwen2-VL."""
        return self.model.get_text_embeddings(texts=query_list)

    def process_image(self, image_list: List[str]):
        """Processes images into embeddings using GME-Qwen2-VL."""
        return self.model.get_image_embeddings(images=image_list)

    def compute_similarity(self, text_features, image_features):
        """ Computes cosine similarity between text and image embeddings. """
        return (text_features @ image_features.T)

    def retrieve(self, query_list: str, image_list: List[str]):
        """
        Retrieves the most relevant images based on the query.

        Args:
            query (str): Query text.
            image_list (list of str): List of image URLs or paths.

        Returns:
            values, indices are sorted similarity results.
        """
        # Compute embeddings
        with torch.no_grad():
            query_embeddings = self.process_text(query_list)
            image_embeddings = self.process_image(image_list)

        similarity_score = self.compute_similarity(query_embeddings, image_embeddings)

        values, indices = torch.sort(similarity_score, dim=1, descending=True)

        return values, indices

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

class CLIPRetriever(BaseRetriever):
    """ CLIP-based retriever for image search using text queries. """

    def __init__(self, checkpoint="ViT-L/14", device="cuda" if torch.cuda.is_available() else "cpu"):
        """ Initializes CLIP model and sets the device. """
        self.device = device
        self.multimodel = True
        self.model, self.preprocess = clip.load(checkpoint, device=self.device)
        self.model.eval()  # Set to evaluation mode

    def process_text(self, text_list, batch_size=1):
        """
        Encodes multiple text queries into feature embeddings using CLIP with batch processing.

        Args:
            text_list (list of str): A list of text queries.
            batch_size (int): Number of text queries to process per batch to avoid OOM.

        Returns:
            torch.Tensor: Normalized text embeddings (shape: [num_texts, embedding_dim]).
        """

        all_embeddings = []

        # Process texts in batches
        for i in range(0, len(text_list)):
            batch_texts = text_list[i]
            end = 77
            while True:
                try:
                    batch_texts = batch_texts[:end]
                    # Tokenize text batch and move to device
                    text_tokens = clip.tokenize(batch_texts).to(self.device)
                    break
                except:
                    end -= 1
                    continue

            with torch.no_grad():
                batch_features = self.model.encode_text(text_tokens)  # Encode text

            # Normalize embeddings
            batch_features = batch_features / batch_features.norm(dim=-1, keepdim=True)

            all_embeddings.append(batch_features.cpu())  # Move to CPU to free up GPU memory

        # Concatenate all embeddings into a single tensor
        return torch.cat(all_embeddings, dim=0)

    def process_image(self, image_paths, batch_size=2):
        """
        Encodes multiple images into feature embeddings using CLIP with batch processing.

        Args:
            image_paths (list of str): A list of image file paths.
            batch_size (int): Number of images to process per batch to avoid OOM.

        Returns:
            torch.Tensor: Normalized image embeddings (shape: [num_images, embedding_dim]).
        """
        valid_images = []
        valid_paths = []

        for image_path in image_paths:
            if os.path.exists(image_path):
                image = Image.open(image_path).convert("RGB")
                valid_images.append(self.preprocess(image))
                valid_paths.append(image_path)  # Store valid paths
            else:
                print(f"Warning: Image file not found - {image_path}")

        if not valid_images:
            return torch.empty(0)  # Return empty tensor if no valid images

        all_embeddings = []

        # Process images in batches
        for i in range(0, len(valid_images), batch_size):
            batch_images = valid_images[i:i + batch_size]

            # Stack images into a batch and move to device
            image_batch = torch.stack(batch_images).to(self.device)

            with torch.no_grad():
                batch_features = self.model.encode_image(image_batch)  # Encode images

            # Normalize embeddings
            batch_features = batch_features / batch_features.norm(dim=-1, keepdim=True)

            all_embeddings.append(batch_features.cpu())  # Move to CPU to manage GPU memory

        # Concatenate all embeddings into a single tensor
        return torch.cat(all_embeddings, dim=0)

    def compute_similarity(self, text_features, image_features):
        """ Computes cosine similarity between text and image embeddings. """
        return (text_features @ image_features.T)

    def retrieve(self, query_list, image_list):
        """
        Retrieves the most relevant images for each text query using CLIP.
        Args:
            query_list (list of str): A list of text queries.
            image_list (list of str): A list of image file paths.
        Returns:
            values, indices are sorted similarity results.
        """
        query_embeddings = self.process_text(query_list)

        # Encode all images once for efficiency
        image_embeddings = self.process_image(image_list)

        similarity_score = self.compute_similarity(query_embeddings, image_embeddings)

        values, indices = torch.sort(similarity_score, dim=1, descending=True)

        return values, indices

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

class SigLIPRetriever(BaseRetriever):
    """ SigLIP-based retriever for image search using text queries. """

    def __init__(self, checkpoint="google/siglip-base-patch16-384", device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initializes SigLIP model, processor, and sets the device.
        
        Args:
            checkpoint (str): The Hugging Face model checkpoint for SigLIP.
            device (str): Device to use ("cuda" or "cpu").
        """
        self.device = device
        self.multimodel = True
        self.processor = AutoProcessor.from_pretrained(checkpoint)  # Preprocessor
        self.model = AutoModel.from_pretrained(checkpoint).to(self.device)  # Load model
        self.model.eval()  # Set to evaluation mode

    def process_text(self, text_list, batch_size=1):
        """
        Encodes multiple text queries into feature embeddings using SigLIP with batch processing.

        Args:
            text_list (list of str): A list of text queries.
            batch_size (int): Number of text queries to process per batch to avoid OOM.

        Returns:
            torch.Tensor: Normalized text embeddings (shape: [num_texts, embedding_dim]).
        """

        all_embeddings = []

        # Process texts in batches
        for i in range(0, len(text_list), 1):
            batch_texts = text_list[i]

            end = 64
            while True:
                try:
                    batch_texts = batch_texts[:end]
                    # Preprocess and encode text
                    inputs = self.processor(text=batch_texts, return_tensors="pt", padding=True).to(self.device)
                    with torch.no_grad():
                        batch_features = self.model.get_text_features(**inputs)  # Get text embeddings
                    break
                except:
                    end -= 1
                    continue

            

            # Normalize embeddings
            batch_features = batch_features / batch_features.norm(dim=-1, keepdim=True)

            all_embeddings.append(batch_features.cpu())  # Move to CPU to free up GPU memory

        # Concatenate all embeddings into a single tensor
        return torch.cat(all_embeddings, dim=0)

    def process_image(self, image_paths, batch_size=2):
        """
        Encodes multiple images into feature embeddings using SigLIP with batch processing.

        Args:
            image_paths (list of str): A list of image file paths.
            batch_size (int): Number of images to process per batch to avoid OOM.

        Returns:
            torch.Tensor: Normalized text embeddings (shape: [num_images, embedding_dim]).
        """
        valid_images = []
        valid_paths = []

        for image_path in image_paths:
            if os.path.exists(image_path):
                image = Image.open(image_path).convert("RGB")
                valid_images.append(image)
                valid_paths.append(image_path)  # Store valid paths
            else:
                print(f"Warning: Image file not found - {image_path}")

        if not valid_images:
            return None  # Return empty dict if no valid images

        all_embeddings = []
        
        # Process images in batches
        for i in range(0, len(valid_images), batch_size):
            batch_images = valid_images[i:i + batch_size]

            # Preprocess and encode images
            inputs = self.processor(images=batch_images, return_tensors="pt").to(self.device)
            with torch.no_grad():
                batch_features = self.model.get_image_features(**inputs)  # Get image embeddings

            # Normalize embeddings
            batch_features = batch_features / batch_features.norm(dim=-1, keepdim=True)

            all_embeddings.append(batch_features.cpu())  # Move to CPU to manage GPU memory

        # Concatenate all embeddings into a single tensor
        return torch.cat(all_embeddings, dim=0)
    
    def compute_similarity(self, text_features, image_features):
        """ Computes cosine similarity between text and image embeddings. """
        return (text_features @ image_features.T)
    
    def retrieve(self, query_list, image_list):
        """
        Retrieves the most relevant images for each text query using SigLIP.

        Args:
            query_list (list of str): A list of text queries.
            image_list (list of str): A list of image file paths.

        Returns:
            values, indices are sorted similarity results.
        """
        query_features = self.process_text(query_list)
        image_features = self.process_image(image_list)

        if not isinstance(image_features, torch.Tensor):
            return None  # If no valid images, return empty dict

        similarity_score = self.compute_similarity(query_features, image_features)

        # Sort similarity scores
        values, indices = torch.sort(similarity_score, dim=1, descending=True)

        return values, indices

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

class NVEmbedRetriever:
    """Retriever class using NVIDIA's NV-Embed-v2 model."""

    def __init__(self, model_name="nvidia/NV-Embed-v2", device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initializes the NV-Embed-v2 model and tokenizer.

        Args:
            model_name (str): The model identifier from Hugging Face.
            device (str, optional): Device to run the model on ('cuda' or 'cpu'). Defaults to GPU if available.
        """

        os.system('pip install transformers==4.46.2')
        self.multimodel = False
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = NVEmbedModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
    
    def add_eos(self, input_examples):
        """
        Adds the end-of-sequence token to input examples.

        Args:
            input_examples (list of str): List of input texts.

        Returns:
            list: List of texts with EOS token appended.
        """
        return [input_example + self.tokenizer.eos_token for input_example in input_examples]

    def retrieve(self, query_list, md_list):
        """
        Performs passage retrieval using NV-Embed-v2.

        Args:
            query_list (list of str): List of query strings.
            md_list (list of str): List of markdown strings.
            batch_size (int): Batch size for encoding.

        Returns:
            values, indices are sorted similarity results.
        """
        # Define instruction for retrieval task
        task_name_to_instruct = {"example": "Given a question, retrieve passages that answer the question"}
        query_prefix = "Instruct: " + task_name_to_instruct["example"] + "\nQuery: "

        # Encode queries and passages
        # get the embeddings
        with torch.no_grad():
            if len(md_list) == 0 or (len(md_list) == 1 and md_list[0] == ''):
                return np.array([]), np.array([])

            query_embeddings = self.model._do_encode(self.add_eos(query_list), batch_size=1, prompt=query_prefix, normalize_embeddings=True)
            passage_embeddings = self.model._do_encode(self.add_eos(md_list), batch_size=1, normalize_embeddings=True)
        
        # Compute similarity scores
        scores = (query_embeddings @ passage_embeddings.T)

        # Rank passages based on similarity
        values, indices = scores.sort(descending=True)

        return values, indices

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

class BM25Retriever:
    """Retriever class using BM25 for document ranking."""

    def __init__(self):
        """Initializes an empty BM25 retriever instance."""
        import nltk
        nltk.download('punkt')

        self.multimodel = False

    def retrieve(self, query_list, md_list):
        """
        Retrieves the most relevant documents for each query using BM25.

        Args:
            query_list (list of str): List of query strings.
            doc_list (list of str): List of document strings.

        Returns:
            list: A list of lists containing indices of top-k relevant documents per query.
            list: A list of lists containing BM25 scores for the top-k retrieved documents per query.
        """
        # Tokenize documents
        tokenized_docs = [word_tokenize(doc.lower()) for doc in md_list]

        if len(tokenized_docs) == 0:
            return np.array([]), np.array([])

        # Initialize BM25 on the given document set
        bm25 = BM25Okapi(tokenized_docs)

        all_indices = []
        all_scores = []

        for query in query_list:
            tokenized_query = word_tokenize(query.lower())
            scores = bm25.get_scores(tokenized_query)

            # Get top-k indices sorted by BM25 scores
            sorted_indices = torch.argsort(torch.tensor(scores), descending=True)
            sorted_scores = [scores[i] for i in sorted_indices]

            all_indices.append(sorted_indices.tolist())
            all_scores.append(sorted_scores)

        return np.array(all_scores), np.array(all_indices)
    
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

class SentenceBERTRetriever:
    """Retriever class using Sentence-BERT for semantic similarity retrieval."""

    def __init__(self, model_name="paraphrase-MiniLM-L6-v2", device=None):
        """
        Initializes the Sentence-BERT model.

        Args:
            model_name (str): The Sentence-BERT model identifier.
            device (str, optional): Device to run the model on ('cuda' or 'cpu'). Defaults to GPU if available.
        """
        self.multimodel = False
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_name).to(self.device)
        self.model.eval()

    def encode_texts(self, texts, batch_size=16):
        """
        Encodes a list of texts into dense embeddings.

        Args:
            texts (list of str): List of input sentences.
            batch_size (int): Batch size for encoding.

        Returns:
            torch.Tensor: Embedding tensor of shape (num_texts, embedding_dim).
        """
        if not texts:
            return torch.empty((0, self.model.get_sentence_embedding_dimension())).to(self.device)

        return self.model.encode(texts, batch_size=batch_size, convert_to_tensor=True, device=self.device, normalize_embeddings=True)

    def retrieve(self, query_list, md_list, top_k=5):
        """
        Performs passage retrieval using Sentence-BERT.

        Args:
            query_list (list of str): List of query strings.
            md_list (list of str): List of passages (documents) to retrieve from.
            top_k (int): Number of top-ranked results to return.

        Returns:
            tuple: (similarity scores, top-ranked passage indices)
        """
        if not md_list or all(md == "" for md in md_list):
            return np.array([]), np.array([])

        # Compute embeddings
        with torch.no_grad():
            query_embeddings = self.encode_texts(query_list)
            passage_embeddings = self.encode_texts(md_list)
    
        # Compute similarity scores
        scores = (query_embeddings @ passage_embeddings.T) * 100

        # Rank passages based on similarity
        values, indices = scores.sort(descending=True)

        return values, indices
    
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

class BGE_M3Retriever:
    """Retriever class using BAAI's BGE-M3 model for text retrieval."""

    def __init__(self, model_name="BAAI/bge-m3", device=None, batch_size=16):
        """
        Initializes the BGE-M3 model and tokenizer.

        Args:
            model_name (str): The model identifier from Hugging Face.
            device (str, optional): Device to run the model on ('cuda' or 'cpu'). Defaults to GPU if available.
            batch_size (int): Batch size for encoding.
        """
        self.multimodel = False
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def encode_texts(self, texts):
        """
        Encodes a list of texts into dense embeddings using batched inference.

        Args:
            texts (list of str): List of input sentences.

        Returns:
            torch.Tensor: Embedding tensor of shape (num_texts, embedding_dim).
        """
        if not texts:
            return torch.empty((0, self.model.config.hidden_size)).to(self.device)

        embeddings_list = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i : i + self.batch_size]
            inputs = self.tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token representation
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)  # Normalize embeddings
                embeddings_list.append(embeddings)

        return torch.cat(embeddings_list, dim=0)

    def retrieve(self, query_list, md_list):
        """
        Performs passage retrieval using BGE-M3 embeddings.

        Args:
            query_list (list of str): List of query strings.
            md_list (list of str): List of passages (documents) to retrieve from.

        Returns:
            tuple: (similarity scores, top-ranked passage indices)
        """
        if not md_list or all(md == "" for md in md_list):
            return np.array([]), np.array([])

        # Compute embeddings in batches
        with torch.no_grad():
            query_embeddings = self.encode_texts(query_list)
            passage_embeddings = self.encode_texts(md_list)

        # Compute similarity scores
        scores = (query_embeddings @ passage_embeddings.T) * 100

        # Rank passages based on similarity
        values, indices = scores.sort(descending=True)

        return values, indices
    
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

class BGE_LargeRetriever:
    """Retriever class using BAAI's BGE-Large model for text retrieval."""

    def __init__(self, model_name="BAAI/bge-large-en", device=None, batch_size=16):
        """
        Initializes the BGE-Large model and tokenizer.

        Args:
            model_name (str): The model identifier from Hugging Face.
            device (str, optional): Device to run the model on ('cuda' or 'cpu'). Defaults to GPU if available.
            batch_size (int): Batch size for encoding.
        """
        self.multimodel = False
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def encode_texts(self, texts):
        """
        Encodes a list of texts into dense embeddings using batched inference.

        Args:
            texts (list of str): List of input sentences.

        Returns:
            torch.Tensor: Embedding tensor of shape (num_texts, embedding_dim).
        """
        if not texts:
            return torch.empty((0, self.model.config.hidden_size)).to(self.device)

        embeddings_list = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i : i + self.batch_size]
            inputs = self.tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token representation
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)  # Normalize embeddings
                embeddings_list.append(embeddings)

        return torch.cat(embeddings_list, dim=0)

    def retrieve(self, query_list, md_list):
        """
        Performs passage retrieval using BGE-Large embeddings.

        Args:
            query_list (list of str): List of query strings.
            md_list (list of str): List of passages (documents) to retrieve from.
            top_k (int): Number of top-ranked results to return.

        Returns:
            tuple: (similarity scores, top-ranked passage indices)
        """
        if not md_list or all(md == "" for md in md_list):
            return np.array([]), np.array([])

        # Compute embeddings in batches
        with torch.no_grad():
            query_embeddings = self.encode_texts(query_list)
            passage_embeddings = self.encode_texts(md_list)


        # Compute similarity scores
        scores = (query_embeddings @ passage_embeddings.T) * 100

        # Rank passages based on similarity
        values, indices = scores.sort(descending=True)

        return values, indices