import math
import torch
from torch import nn
from transformers import T5ForConditionalGeneration
from encoder_pretraining.vqvae_model.vqvae import VQVAE ## Imports in python are so weird man
from transformers.modeling_outputs import BaseModelOutput


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

## Claude suggested this function to avoid crashes, which is honestly a good idea
def load_models(
    checkpoint_path: str = "encoder_pretraining/checkpoints/best.pt",
    t5_name: str = "t5-small",
    freeze_vqvae: bool = True,
) -> "VQVAE_T5":
    """
    Load and return a VQVAE_T5 model.

    Separating instantiation from import prevents crashes when the checkpoint
    is absent and makes it easy to control device/freeze settings from one place.

    Args:
        checkpoint_path: Path to the VQ-VAE checkpoint.
        t5_name:         HuggingFace model name for T5.
        freeze_vqvae:    If True, VQ-VAE weights are frozen during training.
    """
    device = get_device()

    ckpt = torch.load(checkpoint_path, map_location=device)
    vqvae_model = VQVAE(
        h_dim=ckpt["args"]["h_dim"],
        res_h_dim=ckpt["args"]["res_h_dim"],
        n_res_layers=ckpt["args"]["n_res_layers"],
        n_embeddings=ckpt["args"]["n_embeddings"],
        embedding_dim=ckpt["args"]["embedding_dim"],
        beta=ckpt["args"]["beta"],
    ).to(device)
    vqvae_model.load_state_dict(ckpt["model_state_dict"])

    if freeze_vqvae:
        for param in vqvae_model.parameters():
            param.requires_grad = False

    t5_model = T5ForConditionalGeneration.from_pretrained(t5_name)

    return VQVAE_T5(vqvae_model, t5_model).to(device)


## This is from https://github.com/Exorust/TorchLeet/blob/main/v3/modern-architectures/2d-positional-embeddings/2d-positional-embeddings_SOLN.ipynb
def create_2d_sinusoidal_embeddings(height: int, width: int, d_model: int) -> torch.Tensor:
    """
    Create 2D sinusoidal positional embeddings.

    Args:
        height:  number of rows in the patch grid
        width:   number of columns in the patch grid
        d_model: total embedding dimension (must be even)

    Returns:
        Tensor of shape (height * width, d_model)
    """
    assert d_model % 2 == 0, "d_model must be even"
    d_half = d_model // 2

    div_term = torch.exp(
        torch.arange(0, d_half, 2, dtype=torch.float) * -(math.log(10000.0) / d_half)
    )

    row_pos = torch.arange(height, dtype=torch.float).unsqueeze(1)  # (height, 1)
    pe_row = torch.zeros(height, d_half)
    pe_row[:, 0::2] = torch.sin(row_pos * div_term)
    pe_row[:, 1::2] = torch.cos(row_pos * div_term)

    col_pos = torch.arange(width, dtype=torch.float).unsqueeze(1)  # (width, 1)
    pe_col = torch.zeros(width, d_half)
    pe_col[:, 0::2] = torch.sin(col_pos * div_term)
    pe_col[:, 1::2] = torch.cos(col_pos * div_term)

    pe_row = pe_row.unsqueeze(1).expand(height, width, d_half)
    pe_col = pe_col.unsqueeze(0).expand(height, width, d_half)

    pe = torch.cat([pe_row, pe_col], dim=-1)  # (height, width, d_model)
    return pe.reshape(height * width, d_model)


class VQVAE_T5(nn.Module):
    def __init__(self, vqvae_model: VQVAE, t5_model: T5ForConditionalGeneration):
        super().__init__()
        self.vqvae_model = vqvae_model
        self.t5_model = t5_model
        self.d_model = t5_model.config.d_model
        self.image_proj = nn.Linear(vqvae_model.vector_quantization.e_dim, self.d_model)

    def forward(self, x: torch.Tensor, labels: torch.Tensor | None = None):
        z_q, indices, embedding_loss, perplexity = self.vqvae_model.encode_to_tokens(x)
        B, C, H, W = z_q.shape

        # Reshape to (B, H*W, C): T5 expects (batch, seq_len, embed_dim)
        z_q = z_q.permute(0, 2, 3, 1).reshape(B, H * W, C)
        z_q = self.image_proj(z_q)

        pe = create_2d_sinusoidal_embeddings(H, W, self.d_model).to(z_q.device)
        z_q = z_q + pe.unsqueeze(0)

        encoder_outputs = BaseModelOutput(last_hidden_state=z_q)
        t5_output = self.t5_model(encoder_outputs=encoder_outputs, labels=labels)

        return embedding_loss, perplexity, t5_output
    
    @torch.no_grad()
    def generate(self, x: torch.Tensor, **generate_kwargs):
        z_q, indices, embedding_loss, perplexity = self.vqvae_model.encode_to_tokens(x)
        B, C, H, W = z_q.shape
        z_q = z_q.permute(0, 2, 3, 1).reshape(B, H * W, C)
        z_q = self.image_proj(z_q)
        pe = create_2d_sinusoidal_embeddings(H, W, self.d_model).to(z_q.device)
        z_q = z_q + pe.unsqueeze(0)

        encoder_outputs = BaseModelOutput(last_hidden_state=z_q)
        return self.t5_model.generate(encoder_outputs=encoder_outputs, **generate_kwargs)