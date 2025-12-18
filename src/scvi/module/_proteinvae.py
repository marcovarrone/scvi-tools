# New file implementing ProteinVAE for protein-only counts
from __future__ import annotations

from typing import Literal

import torch
from torch import nn
from torch.distributions import Normal
from torch.distributions import kl_divergence as kld

from scvi import REGISTRY_KEYS
from scvi.distributions import NegativeBinomialMixture
from scvi.module._multivae import DecoderADT  # reuse existing protein decoder
from scvi.module.base import BaseModuleClass, LossOutput, auto_move_data
from scvi.nn import Encoder


@auto_move_data
def get_reconstruction_loss_protein(y: torch.Tensor, py_: dict[str, torch.Tensor]):
    """Compute protein reconstruction loss (Negative Binomial Mixture)."""
    py_conditional = NegativeBinomialMixture(
        mu1=py_["rate_back"],
        mu2=py_["rate_fore"],
        theta1=py_["r"],
        mixture_logits=py_["mixing"],
    )
    # Sum over proteins per cell (keep batch dim)
    return -py_conditional.log_prob(y).sum(-1)


class PROTEINVAE(BaseModuleClass):
    """Protein-only variational auto‐encoder.

    This module is a light-weight adaptation of :class:`~scvi.module.TOTALVAE` that drops
    the RNA component and keeps only the protein Negative Binomial mixture likelihood.
    """

    def __init__(
        self,
        n_input_proteins: int,
        n_batch: int = 0,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers_encoder: int = 2,
        n_layers_decoder: int = 2,
        dropout_rate: float = 0.1,
        likelihood: Literal["nb", "lognormal_mixture"] = "nb",
        protein_dispersion: Literal["protein", "protein-batch", "protein-label"] = "protein",
    ) -> None:
        super().__init__()

        self.likelihood = likelihood
        self.n_input_proteins = n_input_proteins
        self.n_batch = n_batch
        self.n_latent = n_latent
        self.protein_dispersion = protein_dispersion

        # Encoder: maps protein counts -> latent z (dimension n_latent)
        self.encoder = Encoder(
            n_input=n_input_proteins,
            n_output=n_latent,
            n_layers=n_layers_encoder,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            distribution="normal",
        )

        # Decoder for proteins (re-uses implementation from MultiVI)
        self.decoder = DecoderADT(
            n_input=n_latent,
            n_output_proteins=n_input_proteins,
            n_layers=n_layers_decoder,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
        )

        # Protein dispersion parameter (theta) – same semantics as TOTALVAE
        if self.protein_dispersion == "protein":
            self.py_r = nn.Parameter(torch.randn(n_input_proteins))
        elif self.protein_dispersion == "protein-batch":
            self.py_r = nn.Parameter(torch.randn(n_input_proteins, n_batch))
        else:  # protein-label (not yet supported, fallback to shared)
            self.py_r = nn.Parameter(torch.randn(n_input_proteins))

    # ---------------------------------------------------------------------
    # Inference / Generative
    # ---------------------------------------------------------------------
    def _get_inference_input(self, tensors):
        return {
            "protein_counts": tensors[REGISTRY_KEYS.PROTEIN_EXP_KEY],
            "batch_index": tensors[REGISTRY_KEYS.BATCH_KEY],
        }

    @auto_move_data
    def inference(self, protein_counts: torch.Tensor, batch_index: torch.Tensor):
        # Encode protein counts into latent space
        qzm, qzv, z = self.encoder(protein_counts, batch_index)
        return {"qzm": qzm, "qzv": qzv, "z": z}

    def _get_generative_input(self, tensors, inference_outputs):
        return {
            "z": inference_outputs["z"],
            "batch_index": tensors[REGISTRY_KEYS.BATCH_KEY],
        }

    @auto_move_data
    def generative(self, z: torch.Tensor, batch_index: torch.Tensor):
        # Decode to protein parameters
        py_, _ = self.decoder(z, batch_index)

        # Dispersion per protein / batch handling
        if self.protein_dispersion == "protein-batch":
            # Map batch index (B,) to (B, n_proteins)
            batch_onehot = torch.nn.functional.one_hot(batch_index.squeeze(-1), self.n_batch).float()
            py_r = torch.exp(torch.matmul(batch_onehot, self.py_r.T))
        else:  # "protein" (shared)
            py_r = torch.exp(self.py_r)
        py_["r"] = py_r
        return {"py_": py_}

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------
    def loss(self, tensors, inference_outputs, generative_outputs, kl_weight: float = 1.0):
        y = tensors[REGISTRY_KEYS.PROTEIN_EXP_KEY]
        py_ = generative_outputs["py_"]

        # Choose reconstruction loss by likelihood
        if self.likelihood == "nb":
            recon_loss = get_reconstruction_loss_protein(y, py_)
        elif self.likelihood == "lognormal_mixture":
            # Mixture of two log-normal components
            from scvi.distributions import LogNormalMixture

            # component1 params: back_alpha, back_beta
            mean1 = py_["back_alpha"]
            sigma1 = py_["back_beta"]
            # component2 params: shift log-mean by log(fore_scale)
            mean2 = mean1 + torch.log(py_["fore_scale"])
            sigma2 = sigma1
            logits = py_["mixing"]
            lnm = LogNormalMixture(mean1, sigma1, mean2, sigma2, logits)
            recon_loss = -lnm.log_prob(y).sum(-1)
        else:
            raise ValueError(f"Unknown likelihood '{self.likelihood}'")

        # KL divergence between q(z|y) and standard Normal prior
        qzm = inference_outputs["qzm"]
        qzv = inference_outputs["qzv"]
        kl_div_z = kld(Normal(qzm, torch.sqrt(qzv)), Normal(0, 1)).sum(-1)

        loss = torch.mean(recon_loss + kl_weight * kl_div_z)

        reconstruction_dict = {"reconstruction_loss_protein": recon_loss}
        kl_local_dict = {"kl_divergence_z": kl_div_z}

        return LossOutput(loss=loss, reconstruction_loss=reconstruction_dict, kl_local=kl_local_dict) 