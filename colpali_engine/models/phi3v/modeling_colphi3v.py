from typing import ClassVar, Optional
from PIL import Image
import torch
from torch import nn
from colpali_engine.models.phi3v.modeling_phi3_v import Phi3VPreTrainedModel, Phi3VModel
from colpali_engine.models.phi3v.image_processing_phi3_v import Phi3VImageProcessor
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoProcessor
from transformers.modeling_utils import PreTrainedModel


class ColPhi(Phi3VPreTrainedModel):

    main_input_name: ClassVar[str] = "doc_input_ids"  # transformers-related
    
    def __init__(self, config):
        self.selected_layer = -12
        super(ColPhi, self).__init__(config)
        model: Phi3VModel = Phi3VModel(config)
        self.model = model
        self.dim = 128
        self.custom_text_proj = nn.Linear(self.model.config.hidden_size, self.dim)
        
        self.param_init()

    def param_init(self):
        # Initialize weights and biases
        nn.init.xavier_normal_(self.custom_text_proj.weight)
        if self.custom_text_proj.bias is not None:
            nn.init.zeros_(self.custom_text_proj.bias)
        
        # Disable gradients for all parameters in the custom_text_proj layer
        for param in self.custom_text_proj.parameters():
            param.requires_grad = False

    def forward(self, *args, **kwargs):
        """
        Forward pass through Llama and the linear layer for dimensionality reduction

        Args:
        - input_ids (torch.LongTensor): The input tokens tensor.
        - attention_mask (torch.LongTensor): The attention mask tensor.

        Returns:
        - torch.Tensor: Embeddings of shape (batch_size, num_tokens, dim)
        """

        kwargs.pop("output_hidden_states", None)
        if "pixel_values" in kwargs:
            kwargs["pixel_values"] = kwargs["pixel_values"].to(dtype=self.dtype)
        
        outputs = self.model(*args, output_hidden_states=True, **kwargs)
        last_hidden_states = outputs.hidden_states[self.selected_layer + 32]  # (batch_size, sequence_length, hidden_size)
        proj = last_hidden_states

        proj = proj / proj.norm(dim=-1, keepdim=True)
        proj = proj * kwargs["attention_mask"].unsqueeze(-1)
        return proj
    