# New file implementing ProteinVI model class for protein-only data
from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Sequence, Number, Iterable

import numpy as np
import pandas as pd
import torch

from scvi import REGISTRY_KEYS
from scvi.data import AnnDataManager, fields
from scvi.model.base import VAEMixin, UnsupervisedTrainingMixin, BaseModelClass, ArchesMixin
from scvi.module import PROTEINVAE
from scvi.utils._docstrings import setup_anndata_dsp

if TYPE_CHECKING:
    from anndata import AnnData


class PROTEINVI(VAEMixin, UnsupervisedTrainingMixin, BaseModelClass, ArchesMixin):
    """Variational inference model for protein counts (ADT) only.

    This is a simplified version of :class:`~scvi.model.TOTALVI` that drops the RNA component.
    """

    _module_cls = PROTEINVAE

    # ------------------------------------------------------------------
    # Model initialization
    # ------------------------------------------------------------------
    def __init__(
        self,
        adata: "AnnData",
        n_latent: int = 10,
        n_hidden: int = 128,
        n_layers_encoder: int = 2,
        n_layers_decoder: int = 2,
        dropout_rate: float = 0.1,
        likelihood: Literal["nb", "lognormal_mixture"] = "nb",
        protein_dispersion: Literal["protein", "protein-batch"] = "protein",
        **module_kwargs,
    ) -> None:
        super().__init__(adata)

        n_batch = self.summary_stats.n_batch
        self.module = self._module_cls(
            n_input_proteins=self.summary_stats.n_proteins,
            n_batch=n_batch,
            n_hidden=n_hidden,
            n_latent=n_latent,
            n_layers_encoder=n_layers_encoder,
            n_layers_decoder=n_layers_decoder,
            dropout_rate=dropout_rate,
            likelihood=likelihood,
            protein_dispersion=protein_dispersion,
            **module_kwargs,
        )

    # ------------------------------------------------------------------
    # AnnData setup helpers
    # ------------------------------------------------------------------
    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
        cls,
        adata: "AnnData",
        protein_expression_obsm_key: str,
        protein_names_uns_key: str | None = None,
        batch_key: str | None = None,
        size_factor_key: str | None = None,
        categorical_covariate_keys: list[str] | None = None,
        continuous_covariate_keys: list[str] | None = None,
        **kwargs,
    ):
        """%(summary)s."""
        setup_method_args = cls._get_setup_method_args(**locals())

        anndata_fields: list[fields.BaseAnnDataField] = [
            fields.CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
            fields.NumericalObsField(
                REGISTRY_KEYS.SIZE_FACTOR_KEY,
                size_factor_key,
                required=False,
            ),
            fields.CategoricalJointObsField(
                REGISTRY_KEYS.CAT_COVS_KEY,
                categorical_covariate_keys,
            ),
            fields.NumericalJointObsField(
                REGISTRY_KEYS.CONT_COVS_KEY,
                continuous_covariate_keys,
            ),
            fields.ProteinObsmField(
                REGISTRY_KEYS.PROTEIN_EXP_KEY,
                protein_expression_obsm_key,
                use_batch_mask=False,
                colnames_uns_key=protein_names_uns_key,
                is_count_data=False,
            ),
        ]

        adata_manager = AnnDataManager(fields=anndata_fields, setup_method_args=setup_method_args)
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)

    @torch.inference_mode()
    def get_normalized_expression(
        self,
        adata=None,
        indices=None,
        n_samples_overall: int | None = None,
        transform_batch: Sequence[Number | str] | None = None,
        protein_list: Sequence[str] | None = None,
        n_samples: int = 1,
        sample_protein_mixing: bool = False,
        scale_protein: bool = False,
        include_protein_background: bool = False,
        batch_size: int | None = None,
        return_mean: bool = True,
        return_numpy: bool | None = None,
        silent: bool = True,
    ) -> np.ndarray | pd.DataFrame:
        r"""Returns the normalized protein expression.

        This is the denoised protein expression, computed as 
        :math:`(1-\pi_{nt})\alpha_{nt}\beta_{nt}` where :math:`\pi_{nt}` is the 
        probability of background, :math:`\alpha_{nt}` is the foreground scaling, 
        and :math:`\beta_{nt}` is the background mean.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        n_samples_overall
            Number of samples to use in total
        transform_batch
            Batch to condition on.
            If transform_batch is:

            - None, then real observed batch is used
            - int, then batch transform_batch is used
            - List[int], then average over batches in list
        protein_list
            Return protein expression for a subset of proteins.
            This can save memory when working with large datasets and few proteins are
            of interest.
        n_samples
            Get sample scale from multiple samples.
        sample_protein_mixing
            Sample mixing bernoulli, setting background to zero
        scale_protein
            Make protein expression sum to 1
        include_protein_background
            Include background component for protein expression
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        return_mean
            Whether to return the mean of the samples.
        return_numpy
            Return a `np.ndarray` instead of a `pd.DataFrame`. Includes protein
            names as columns. If either n_samples=1 or return_mean=True, defaults to False.
            Otherwise, it defaults to True.
        silent
            Whether to disable progress bar.

        Returns
        -------
        **protein_normalized_expression** - normalized expression for proteins

        If ``n_samples`` > 1 and ``return_mean`` is False, then the shape is
        ``(samples, cells, proteins)``. Otherwise, shape is ``(cells, proteins)``. Return type is
        ``pd.DataFrame`` unless ``return_numpy`` is True.
        """
        from scvi.data import _constants
        from scvi.data._utils import _get_var_names_from_manager
        from scvi.settings import batch_size as default_batch_size
        from scvi.utils._docstrings import de_dsp
        from scvi.utils._utils import _get_batch_code_from_category
        from scvi.utils import track

        adata = self._validate_anndata(adata)
        adata_manager = self.get_anndata_manager(adata)
        if indices is None:
            indices = np.arange(adata.n_obs)
        if n_samples_overall is not None:
            indices = np.random.choice(indices, n_samples_overall)
        post = self._make_data_loader(adata=adata, indices=indices, batch_size=batch_size)

        if protein_list is None:
            protein_mask = slice(None)
        else:
            all_proteins = self.protein_state_registry.column_names
            protein_mask = [True if p in protein_list else False for p in all_proteins]

        if n_samples > 1 and return_mean is False:
            if return_numpy is False:
                import warnings
                warnings.warn(
                    "`return_numpy` must be `True` if `n_samples > 1` and `return_mean` "
                    "is `False`, returning an `np.ndarray`.",
                    UserWarning,
                    stacklevel=2,
                )
            return_numpy = True

        if not isinstance(transform_batch, Iterable):
            transform_batch = [transform_batch]

        transform_batch = _get_batch_code_from_category(adata_manager, transform_batch)

        scale_list_pro = []

        for tensors in track(post, description="Computing normalized protein expression"):
            y = tensors[REGISTRY_KEYS.PROTEIN_EXP_KEY]
            py_scale = torch.zeros_like(y)[..., protein_mask]
            if n_samples > 1:
                py_scale = torch.stack(n_samples * [py_scale])
            
            for b in track(transform_batch, disable=silent):
                generative_kwargs = {"transform_batch": b}
                inference_kwargs = {"n_samples": n_samples}
                _, generative_outputs = self.module.forward(
                    tensors=tensors,
                    inference_kwargs=inference_kwargs,
                    generative_kwargs=generative_kwargs,
                    compute_loss=False,
                )

                py_ = generative_outputs["py_"]
                # probability of background
                protein_mixing = 1 / (1 + torch.exp(-py_["mixing"].cpu()))
                if sample_protein_mixing is True:
                    protein_mixing = torch.distributions.Bernoulli(protein_mixing).sample()
                protein_val = py_["rate_fore"].cpu() * (1 - protein_mixing)
                if include_protein_background is True:
                    protein_val += py_["rate_back"].cpu() * protein_mixing

                if scale_protein is True:
                    protein_val = torch.nn.functional.normalize(protein_val, p=1, dim=-1)
                protein_val = protein_val[..., protein_mask]
                py_scale += protein_val
            
            py_scale /= len(transform_batch)
            scale_list_pro.append(py_scale)

        if n_samples > 1:
            # concatenate along batch dimension -> result shape = (samples, cells, features)
            scale_list_pro = torch.cat(scale_list_pro, dim=1)
            # (cells, features, samples)
            scale_list_pro = scale_list_pro.permute(1, 2, 0)
        else:
            scale_list_pro = torch.cat(scale_list_pro, dim=0)

        if return_mean is True and n_samples > 1:
            scale_list_pro = torch.mean(scale_list_pro, dim=-1)

        scale_list_pro = scale_list_pro.cpu().numpy()
        
        if return_numpy is None or return_numpy is False:
            protein_names = self.protein_state_registry.column_names
            pro_df = pd.DataFrame(
                scale_list_pro,
                columns=protein_names[protein_mask],
                index=adata.obs_names[indices],
            )
            return pro_df
        else:
            return scale_list_pro 