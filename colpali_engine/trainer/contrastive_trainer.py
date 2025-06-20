import torch
from transformers import Trainer
import torch
import os
from typing import Optional

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

class ContrastiveTrainer_InternVL2(Trainer):
    def __init__(self, loss_func, is_vision_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_func = loss_func
        self.is_vision_model = is_vision_model

    def encode_img(self, model, batch_image):
        all_embeddings = []
        for meta in batch_image:
            meta = {k: v.to('cuda') for k, v in meta.items()}
            embeddings = model(**meta)
            all_embeddings.append(embeddings)
    
        all_embeddings = pad_and_cat_tensors(all_embeddings)
        
        return all_embeddings

    def encode_text(self, model, batch_query):
        all_embeddings = []
        for meta in batch_query:
            meta = {k: v.to('cuda') for k, v in meta.items()}
            embeddings = model(**meta)
            all_embeddings.append(embeddings)

        all_embeddings = pad_and_cat_tensors(all_embeddings)
        
        return all_embeddings
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        [batch_query, batch_image, neg_batch_image] = inputs
        
        doc_outputs = self.encode_img(model, batch_image)
        query_outputs = self.encode_text(model, batch_query)

        if neg_batch_image is not None:
            neg_doc_outputs = self.encode_img(model, neg_batch_image)
            loss = self.loss_func(query_outputs, doc_outputs, neg_doc_outputs)
            return (loss, (query_outputs, doc_outputs, neg_doc_outputs)) if return_outputs else loss

        loss = self.loss_func(query_outputs, doc_outputs)
        return (loss, (query_outputs, doc_outputs)) if return_outputs else loss
        

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=True):
        
        [batch_query, batch_image, neg_batch_image] = inputs
        
        """This function is used to generate predictions and return the loss for the given inputs."""
        if not prediction_loss_only:
            raise ValueError("prediction_step is only called with prediction_loss_only=True")

        with torch.no_grad():
            doc_outputs = self.encode_img(model, batch_image)
            query_outputs = self.encode_text(model, batch_query)

            if neg_batch_image is not None:
                neg_doc_outputs = self.encode_img(model, neg_batch_image)
                loss = self.loss_func(query_outputs, doc_outputs, neg_doc_outputs)
            else:            
               loss = self.loss_func(query_outputs, doc_outputs)

            return loss, None, None



class ContrastiveTrainer(Trainer):
    def __init__(self, loss_func, is_vision_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_func = loss_func
        self.is_vision_model = is_vision_model  # Unused argument, will be removed in 0.4.0

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        query_outputs = model(input_ids=inputs["query_input_ids"], attention_mask=inputs["query_attention_mask"])
        # feed only kwargs with 'doc_' prefix
        doc_outputs = model(**{k[4:]: v for k, v in inputs.items() if k.startswith("doc")})
        if "neg_doc_input_ids" in inputs:
            neg_doc_outputs = model(**{k[8:]: v for k, v in inputs.items() if k.startswith("neg_doc")})
            loss = self.loss_func(query_outputs, doc_outputs, neg_doc_outputs)
            return (loss, (query_outputs, doc_outputs, neg_doc_outputs)) if return_outputs else loss

        loss = self.loss_func(query_outputs, doc_outputs)
        return (loss, (query_outputs, doc_outputs)) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=True):
        """This function is used to generate predictions and return the loss for the given inputs."""
        if not prediction_loss_only:
            raise ValueError("prediction_step is only called with prediction_loss_only=True")

        with torch.no_grad():
            # feed only kwargs with 'doc_' prefix
            doc_outputs = model(**{k[4:]: v for k, v in inputs.items() if k.startswith("doc")})
            query_outputs = model(input_ids=inputs["query_input_ids"], attention_mask=inputs["query_attention_mask"])
            if "neg_doc_input_ids" in inputs:
                neg_doc_outputs = model(**{k[8:]: v for k, v in inputs.items() if k.startswith("neg_doc")})
                loss = self.loss_func(query_outputs, doc_outputs, neg_doc_outputs)
                return loss, None, None

            loss = self.loss_func(query_outputs, doc_outputs)
            return loss, None, None
