from typing import ClassVar, List, Optional
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import torch
from torch import nn
from colpali_engine.models.InternVL2.model_4b_util.modeling_internvl_chat import InternVLChatModel_4B, InternVL2PreTrainedModel_4B


class ColInternvl2_4b(InternVL2PreTrainedModel_4B):
    """
    ColQwen2 model implementation from the "ColInternvl2_4b: Efficient Document Retrieval with Vision Language Models" paper.
    """

    main_input_name: ClassVar[str] = "doc_input_ids"  # transformers-related
    
    def __init__(self, config):
        super(ColInternvl2_4b, self).__init__(config)

        self.model: InternVLChatModel_4B = InternVLChatModel_4B(config)
        
        self.dim = 128
        self.main_input_name = "doc_input_ids"

        self.custom_text_proj = nn.Linear(config.llm_config.hidden_size, self.dim)

        self.param_init()

    def param_init(self):
        # Initialize weights and biases
        nn.init.xavier_normal_(self.custom_text_proj.weight)
        if self.custom_text_proj.bias is not None:
            nn.init.zeros_(self.custom_text_proj.bias)
        
        # Disable gradients for all parameters in the custom_text_proj layer
        for param in self.custom_text_proj.parameters():
            param.requires_grad = False

    def load_custom_proj(self, model_name):
        """
        Loads the custom projection layer weights from a Hugging Face model repository.

        Args:
            model_name (str): The Hugging Face model ID (e.g., "your-username/custom-model").
        """
        try:
            # Download the safetensors file from the Hugging Face repository
            custom_proj_path = hf_hub_download(repo_id=model_name, filename="custom_text_proj.safetensors")

            # Load the weights from safetensors
            loaded_state_dict = load_file(custom_proj_path)

            # Assign weights to the custom projection layer
            self.custom_text_proj.weight = torch.nn.Parameter(loaded_state_dict['base_layer.weight'].clone().to(self.device))
            self.custom_text_proj.bias = torch.nn.Parameter(loaded_state_dict['base_layer.bias'].clone().to(self.device))

            print(f"✅ Successfully loaded custom_text_proj from {custom_proj_path}")

        except Exception as e:
            print(f"⚠️ Warning: Failed to load custom_text_proj.safetensors from {model_name}. Error: {e}")

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """
        Custom from_pretrained() method to load both model weights and custom projection layer.
        """
        # Load the base model
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

        # Load the custom projection layer
        model.load_custom_proj(pretrained_model_name_or_path)

        return model


    def forward(self, *args, **kwargs):
        """
        Forward pass through Llama and the linear layer for dimensionality reduction

        Args:
        - input_ids (torch.LongTensor): The input tokens tensor.
        - attention_mask (torch.LongTensor): The attention mask tensor.

        Returns:
        - torch.Tensor: Embeddings of shape (batch_size, num_tokens, dim)
        """

        kwargs['output_hidden_states'] = True
        if 'pixel_values' in kwargs.keys():
            outputs = self.model(*args, **kwargs)
        else:
            # outputs = self.model.language_model(*args, output_hidden_states=True, **kwargs)
            outputs = self.model.language_model(*args, **kwargs)
        
        last_hidden_states = outputs.hidden_states[-1]  # (batch_size, sequence_length, hidden_size)
        
        proj = self.custom_text_proj(last_hidden_states)
        
        # normalize l2 norm
        proj = proj / proj.norm(dim=-1, keepdim=True)
        
        return proj

