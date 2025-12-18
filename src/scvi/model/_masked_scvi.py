"""MaskedSCVI model for gene imputation with missing genes."""

from __future__ import annotations

import logging
import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from scvi import REGISTRY_KEYS, settings
from scvi.data._constants import ADATA_MINIFY_TYPE
from scvi.model._scvi import SCVI
from scvi.model._utils import _init_library_size
from scvi.module._masked_vae import MaskedVAE

if TYPE_CHECKING:
    from typing import Literal

    from anndata import AnnData

logger = logging.getLogger(__name__)


class MaskedSCVI(SCVI):
    """scVI model with gene masking for imputation.

    Extends :class:`~scvi.model.SCVI` to allow masking certain genes from the
    reconstruction loss on a per-batch basis. This is useful when some batches
    are missing certain genes and you want to impute them.

    Parameters
    ----------
    adata
        AnnData object that has been registered via :meth:`~scvi.model.SCVI.setup_anndata`.
    gene_batch_mask
        A dictionary mapping batch names (as they appear in the `batch_key` column) to a
        sequence of gene names that should be *excluded* (masked) from the
        reconstruction loss for that batch.
    **kwargs
        Keyword arguments for :class:`~scvi.model.SCVI`.

    Examples
    --------
    >>> MaskedSCVI.setup_anndata(adata, batch_key="batch")
    >>> gene_batch_mask = {"batch_A": ["Gene1", "Gene2"]}  # Mask Gene1 and Gene2 for batch_A
    >>> model = MaskedSCVI(adata, gene_batch_mask=gene_batch_mask)
    >>> model.train()
    >>> adata.obsm["X_scVI"] = model.get_latent_representation()
    """

    _module_cls = MaskedVAE

    def __init__(
        self,
        adata: AnnData | None = None,
        registry: dict | None = None,
        gene_batch_mask: dict[str, Sequence[str]] | None = None,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        dispersion: Literal["gene", "gene-batch", "gene-label", "gene-cell"] = "gene",
        gene_likelihood: Literal["zinb", "nb", "poisson", "normal"] = "zinb",
        use_observed_lib_size: bool = True,
        latent_distribution: Literal["normal", "ln"] = "normal",
        **kwargs,
    ):
        # Call grandparent init to set up adata_manager without creating module
        from scvi.model.base import BaseMinifiedModeModelClass

        BaseMinifiedModeModelClass.__init__(self, adata, registry)

        self._module_kwargs = {
            "n_hidden": n_hidden,
            "n_latent": n_latent,
            "n_layers": n_layers,
            "dropout_rate": dropout_rate,
            "dispersion": dispersion,
            "gene_likelihood": gene_likelihood,
            "latent_distribution": latent_distribution,
            **kwargs,
        }
        self._model_summary_string = (
            "MaskedSCVI model with the following parameters: \n"
            f"n_hidden: {n_hidden}, n_latent: {n_latent}, n_layers: {n_layers}, "
            f"dropout_rate: {dropout_rate}, dispersion: {dispersion}, "
            f"gene_likelihood: {gene_likelihood}, latent_distribution: {latent_distribution}."
        )

        if self._module_init_on_train:
            self.module = None
            warnings.warn(
                "Model was initialized without `adata`. The module will be initialized when "
                "calling `train`. This behavior is experimental and may change in the future.",
                UserWarning,
                stacklevel=settings.warnings_stacklevel,
            )
        else:
            # Process gene_batch_mask
            processed_mask = None
            if gene_batch_mask is not None:
                processed_mask = self._process_gene_batch_mask(gene_batch_mask)

            n_cats_per_cov = (
                self.adata_manager.get_state_registry(
                    REGISTRY_KEYS.CAT_COVS_KEY
                ).n_cats_per_key
                if REGISTRY_KEYS.CAT_COVS_KEY in self.adata_manager.data_registry
                else None
            )

            n_batch = self.summary_stats.n_batch
            use_size_factor_key = self.registry_["setup_args"][
                f"{REGISTRY_KEYS.SIZE_FACTOR_KEY}_key"
            ]
            library_log_means, library_log_vars = None, None
            if (
                not use_size_factor_key
                and self.minified_data_type != ADATA_MINIFY_TYPE.LATENT_POSTERIOR
                and not use_observed_lib_size
            ):
                library_log_means, library_log_vars = _init_library_size(
                    self.adata_manager, n_batch
                )

            self.module = self._module_cls(
                n_input=self.summary_stats.n_vars,
                n_batch=n_batch,
                n_labels=self.summary_stats.n_labels,
                n_continuous_cov=self.summary_stats.get("n_extra_continuous_covs", 0),
                n_cats_per_cov=n_cats_per_cov,
                n_hidden=n_hidden,
                n_latent=n_latent,
                n_layers=n_layers,
                dropout_rate=dropout_rate,
                dispersion=dispersion,
                gene_likelihood=gene_likelihood,
                use_observed_lib_size=use_observed_lib_size,
                latent_distribution=latent_distribution,
                use_size_factor_key=use_size_factor_key,
                library_log_means=library_log_means,
                library_log_vars=library_log_vars,
                gene_batch_mask=processed_mask,
                **kwargs,
            )
            self.module.minified_data_type = self.minified_data_type

        self.init_params_ = self._get_init_params(locals())

    def _process_gene_batch_mask(
        self, gene_batch_mask: dict[str, Sequence[str]]
    ) -> dict[str, np.ndarray]:
        """Convert user-friendly gene_batch_mask to internal format.

        Parameters
        ----------
        gene_batch_mask
            Dictionary mapping batch names to gene names to mask.

        Returns
        -------
        Dictionary mapping batch indices (as strings) to boolean mask arrays.
        """
        batch_state_registry = self.adata_manager.get_state_registry(REGISTRY_KEYS.BATCH_KEY)
        categorical_mapping = batch_state_registry.categorical_mapping
        var_names = self.adata_manager.adata.var_names

        processed_mask = {}
        for batch_name, genes_to_mask in gene_batch_mask.items():
            # Find batch index
            batch_idx = None
            for i, name in enumerate(categorical_mapping):
                if name == batch_name:
                    batch_idx = i
                    break
            if batch_idx is None:
                raise ValueError(
                    f"Batch name '{batch_name}' not found in registered batches: "
                    f"{list(categorical_mapping)}"
                )

            # Create mask: 1 for genes to keep, 0 for genes to mask
            mask = np.ones(len(var_names), dtype=np.float32)
            for gene in genes_to_mask:
                if gene in var_names:
                    gene_idx = var_names.get_loc(gene)
                    mask[gene_idx] = 0.0
                else:
                    warnings.warn(
                        f"Gene '{gene}' not found in var_names and will be ignored.",
                        UserWarning,
                        stacklevel=settings.warnings_stacklevel,
                    )

            processed_mask[str(batch_idx)] = mask

        return processed_mask
