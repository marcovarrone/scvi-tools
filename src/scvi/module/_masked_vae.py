"""MaskedVAE module for gene imputation with missing genes."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from scvi import REGISTRY_KEYS
from scvi.module._constants import MODULE_KEYS
from scvi.module._vae import VAE
from scvi.module.base import LossOutput
from scvi.utils import unsupported_if_adata_minified

if TYPE_CHECKING:
    from torch.distributions import Distribution


class MaskedVAE(VAE):
    """Variational auto-encoder with gene masking for imputation.

    Extends :class:`~scvi.module.VAE` to allow masking certain genes from the
    reconstruction loss on a per-batch basis.

    Parameters
    ----------
    gene_batch_mask
        A dictionary mapping batch index (as string, e.g., "0", "1") to a numpy
        array of shape ``(n_genes,)`` with 1s for genes to include in the loss
        and 0s for genes to exclude (mask).
    **kwargs
        Keyword arguments for :class:`~scvi.module.VAE`.
    """

    def __init__(
        self,
        gene_batch_mask: dict[str, np.ndarray] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.gene_batch_mask = gene_batch_mask

    @unsupported_if_adata_minified
    def loss(
        self,
        tensors: dict[str, torch.Tensor],
        inference_outputs: dict[str, torch.Tensor | Distribution | None],
        generative_outputs: dict[str, Distribution | None],
        kl_weight: torch.tensor | float = 1.0,
    ) -> LossOutput:
        """Compute the loss with optional gene masking."""
        from torch.distributions import kl_divergence

        x = tensors[REGISTRY_KEYS.X_KEY]
        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]

        kl_divergence_z = kl_divergence(
            inference_outputs[MODULE_KEYS.QZ_KEY], generative_outputs[MODULE_KEYS.PZ_KEY]
        ).sum(dim=-1)

        if not self.use_observed_lib_size:
            kl_divergence_l = kl_divergence(
                inference_outputs[MODULE_KEYS.QL_KEY], generative_outputs[MODULE_KEYS.PL_KEY]
            ).sum(dim=1)
        else:
            kl_divergence_l = torch.zeros_like(kl_divergence_z)

        # Per-gene reconstruction loss (NLL)
        reconst_loss_per_gene = -generative_outputs[MODULE_KEYS.PX_KEY].log_prob(x)

        # Apply gene batch mask if provided
        if self.gene_batch_mask is not None:
            gene_mask_minibatch = torch.ones_like(reconst_loss_per_gene)
            for b in torch.unique(batch_index):
                b_key = str(int(b.item()))
                if b_key in self.gene_batch_mask:
                    b_indices = (batch_index == b).reshape(-1)
                    gene_mask_minibatch[b_indices] = torch.tensor(
                        self.gene_batch_mask[b_key].astype(np.float32),
                        device=x.device,
                    )
            reconst_loss_per_gene = reconst_loss_per_gene * gene_mask_minibatch

        reconst_loss = reconst_loss_per_gene.sum(-1)

        kl_local_for_warmup = kl_divergence_z
        kl_local_no_warmup = kl_divergence_l

        weighted_kl_local = kl_weight * kl_local_for_warmup + kl_local_no_warmup

        loss = torch.mean(reconst_loss + weighted_kl_local)

        if self.extra_payload_autotune:
            extra_metrics_payload = {
                "z": inference_outputs["z"],
                "batch": tensors[REGISTRY_KEYS.BATCH_KEY],
                "labels": tensors[REGISTRY_KEYS.LABELS_KEY],
            }
        else:
            extra_metrics_payload = {}

        return LossOutput(
            loss=loss,
            reconstruction_loss=reconst_loss,
            kl_local={
                MODULE_KEYS.KL_L_KEY: kl_divergence_l,
                MODULE_KEYS.KL_Z_KEY: kl_divergence_z,
            },
            extra_metrics=extra_metrics_payload,
        )
