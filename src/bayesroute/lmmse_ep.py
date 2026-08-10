from __future__ import annotations

"""Damped extrinsic LMMSE message passing for BayesRoute-Rx.

This module replaces the scalar-variance soft-PIC front end used in Gate-1.
For each target stream and resource element, it forms a spatial LMMSE filter
from the posterior channel mean. Graph-selected strong interferers are softly
cancelled; omitted interferers remain zero-mean Gaussian terms. The target's
own previous belief is excluded from its message, which makes the update
extrinsic. Discrete QAM moment matching and damping close the loop.
"""

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from .qam import (
    make_qam_constellation,
    symbol_logits_to_bit_logits,
    symbol_probs_to_mean_var,
)


DETECTOR_VERSION = "damped_extrinsic_lmmse_v1"


@dataclass
class LMMSEEPDiagnostics:
    min_extrinsic_variance: float
    max_extrinsic_variance: float
    min_candidate_variance: float
    max_candidate_variance: float
    min_filter_gain: float
    max_filter_gain: float
    min_belief_variance: float
    max_belief_variance: float
    max_hermitian_error: float
    finite: bool


def _noise_grid(
    noise_var: torch.Tensor | float,
    *,
    batch_size: int,
    n_data: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    value = torch.as_tensor(noise_var, device=device).real.to(dtype)
    if value.ndim == 0:
        return value.expand(batch_size, n_data)
    if value.ndim == 1 and value.shape[0] == batch_size:
        return value[:, None].expand(batch_size, n_data)
    if value.ndim == 2 and value.shape == (batch_size, n_data):
        return value
    return value.mean().expand(batch_size, n_data)


def _local_covariance_data(
    local_cov: torch.Tensor,
    data_idx: torch.Tensor,
    *,
    batch_size: int,
) -> torch.Tensor:
    """Return [B,D,N,N] local channel-error covariance."""
    if local_cov.ndim == 3:
        # [N,N,R] -> [B,D,N,N]
        selected = local_cov[..., data_idx]
        return selected.permute(2, 0, 1)[None, ...].expand(batch_size, -1, -1, -1)
    if local_cov.ndim == 4:
        # [B,N,N,R] -> [B,D,N,N]
        if local_cov.shape[0] != batch_size:
            raise ValueError("Batch-dependent local covariance has wrong batch size")
        return local_cov[..., data_idx].permute(0, 3, 1, 2)
    raise ValueError("local_cov must have shape [N,N,R] or [B,N,N,R]")


def full_directed_graph(
    batch_size: int,
    n_data: int,
    n_streams: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    eye = torch.eye(n_streams, dtype=torch.bool, device=device).view(
        1, 1, n_streams, n_streams
    )
    return torch.ones(
        (batch_size, n_data, n_streams, n_streams),
        dtype=torch.bool,
        device=device,
    ) & (~eye)


def _channel_error_terms(
    covariance: torch.Tensor,
    symbol_mean: torch.Tensor,
    symbol_var: torch.Tensor,
    strong: torch.Tensor,
    target: int,
    *,
    covariance_mode: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return non-target power, target variance, and target cross term.

    All three outputs have shape ``[B,D]``. For one receive antenna, the
    candidate-conditioned channel-error variance is

    ``other_power + |a|^2 target_variance + 2 Re{a target_cross}``.

    The target is excluded from its own incoming message. Strong neighbours
    use their current discrete moments. Omitted neighbours retain zero mean
    and unit QAM energy.
    """
    batch_size, n_data, n_streams, _ = covariance.shape
    if covariance_mode == "none":
        zero_real = torch.zeros(
            (batch_size, n_data),
            dtype=covariance.real.dtype,
            device=covariance.device,
        )
        zero_complex = torch.zeros(
            (batch_size, n_data),
            dtype=covariance.dtype,
            device=covariance.device,
        )
        return zero_real, zero_real, zero_complex

    diagonal = torch.diagonal(covariance, dim1=-2, dim2=-1).real.clamp_min(0.0)
    target_variance = diagonal[..., target]
    other_mask = torch.ones_like(strong, dtype=torch.bool)
    other_mask[..., target] = False

    # Strong neighbours use their current second moments. Weak neighbours are
    # not cancelled, so their cavity mean is zero and their residual energy is
    # the unit-power QAM prior.
    second = symbol_var + torch.abs(symbol_mean) ** 2
    other_energy = torch.where(strong, second, torch.ones_like(second))
    other_energy = torch.where(
        other_mask, other_energy, torch.zeros_like(other_energy)
    )

    if covariance_mode == "diagonal":
        other_power = torch.sum(diagonal * other_energy, dim=-1)
        target_cross = torch.zeros(
            (batch_size, n_data),
            dtype=covariance.dtype,
            device=covariance.device,
        )
        return (
            other_power.real.clamp_min(0.0),
            target_variance.real.clamp_min(0.0),
            target_cross,
        )
    if covariance_mode != "full":
        raise ValueError("covariance_mode must be one of: none, diagonal, full")

    means = torch.where(strong, symbol_mean, torch.zeros_like(symbol_mean))
    means[..., target] = 0.0
    residual_variance = torch.where(
        strong, symbol_var, torch.ones_like(symbol_var)
    )
    residual_variance[..., target] = 0.0
    mean_quadratic = torch.einsum(
        "bdn,bdnm,bdm->bd",
        means,
        covariance,
        means.conj(),
    ).real
    diagonal_term = torch.sum(diagonal * residual_variance, dim=-1)
    other_power = (mean_quadratic + diagonal_term).real.clamp_min(0.0)
    target_cross = torch.einsum(
        "bdm,bdm->bd", covariance[..., target, :], means.conj()
    )
    return other_power, target_variance.real.clamp_min(0.0), target_cross


class DampedExtrinsicLMMSEDetector(nn.Module):
    """Graph-routed, uncertainty-aware, damped extrinsic LMMSE detector.

    Parameters
    ----------
    bits_per_symbol:
        QAM bits per symbol.
    n_iter:
        Number of simultaneous extrinsic updates.
    damping:
        Moment damping in [0,1]. A value of one accepts each new discrete
        moment fully. Values around 0.35--0.7 are screened in Gate-1.
    covariance_mode:
        ``"diagonal"`` uses marginal posterior channel variances,
        ``"full"`` also uses cross-stream posterior covariance, and ``"none"``
        disables channel uncertainty.
    """

    def __init__(
        self,
        bits_per_symbol: int,
        *,
        n_iter: int = 4,
        damping: float = 0.5,
        covariance_mode: str = "diagonal",
        jitter: float = 1e-5,
        variance_floor: float = 1e-6,
        variance_ceiling: float = 4.0,
    ) -> None:
        super().__init__()
        if not 0.0 <= float(damping) <= 1.0:
            raise ValueError("damping must lie in [0,1]")
        if int(n_iter) < 1:
            raise ValueError("n_iter must be positive")
        self.bits_per_symbol = int(bits_per_symbol)
        self.n_iter = int(n_iter)
        self.damping = float(damping)
        self.covariance_mode = str(covariance_mode)
        self.jitter = float(jitter)
        self.variance_floor = float(variance_floor)
        self.variance_ceiling = float(variance_ceiling)

    def forward(
        self,
        y: torch.Tensor,
        h_mean: torch.Tensor,
        local_cov: torch.Tensor,
        data_idx: torch.Tensor,
        noise_var: torch.Tensor | float,
        graph_mask: torch.Tensor | None = None,
        *,
        n_iter: int | None = None,
        damping: float | None = None,
        covariance_mode: str | None = None,
    ) -> dict[str, Any]:
        device = y.device
        iterations = self.n_iter if n_iter is None else int(n_iter)
        beta = self.damping if damping is None else float(damping)
        mode = self.covariance_mode if covariance_mode is None else str(covariance_mode)
        if iterations < 1:
            raise ValueError("n_iter must be positive")
        if not 0.0 <= beta <= 1.0:
            raise ValueError("damping must lie in [0,1]")

        constellation, bit_table = make_qam_constellation(
            self.bits_per_symbol, device=device
        )
        constellation_energy = torch.mean(torch.abs(constellation) ** 2).real

        y_data = y[..., data_idx].permute(0, 2, 1).contiguous()  # [B,D,RX]
        h_data = h_mean[..., data_idx].permute(0, 3, 2, 1).contiguous()  # [B,D,RX,N]
        batch_size, n_data, n_rx, n_streams = h_data.shape
        covariance = _local_covariance_data(
            local_cov, data_idx, batch_size=batch_size
        ).to(device=device, dtype=h_data.dtype)
        if covariance.shape != (batch_size, n_data, n_streams, n_streams):
            raise ValueError("local covariance shape is incompatible with h_mean")

        if graph_mask is None:
            graph_mask = full_directed_graph(
                batch_size, n_data, n_streams, device=device
            )
        if graph_mask.shape != (batch_size, n_data, n_streams, n_streams):
            raise ValueError("graph_mask has an incompatible shape")
        graph_mask = graph_mask.bool()
        diagonal_graph = torch.diagonal(graph_mask, dim1=-2, dim2=-1)
        if diagonal_graph.any():
            raise ValueError("graph_mask must not contain self-edges")

        noise = _noise_grid(
            noise_var,
            batch_size=batch_size,
            n_data=n_data,
            device=device,
            dtype=y_data.real.dtype,
        ).clamp_min(self.variance_floor)

        symbol_mean = torch.zeros(
            (batch_size, n_data, n_streams),
            dtype=h_data.dtype,
            device=device,
        )
        symbol_var = torch.full(
            (batch_size, n_data, n_streams),
            float(constellation_energy.item()),
            dtype=y_data.real.dtype,
            device=device,
        )

        identity = torch.eye(n_rx, dtype=h_data.dtype, device=device).view(
            1, 1, n_rx, n_rx
        )
        last_logits: torch.Tensor | None = None
        last_z = torch.zeros_like(symbol_mean)
        last_v = torch.ones_like(symbol_var)
        min_gain = torch.tensor(float("inf"), device=device)
        max_gain = torch.tensor(0.0, device=device)
        max_hermitian = torch.tensor(0.0, device=device)
        min_candidate_variance = torch.tensor(float("inf"), device=device)
        max_candidate_variance = torch.tensor(0.0, device=device)

        stream_index = torch.arange(n_streams, device=device)
        for _ in range(iterations):
            logits_by_stream: list[torch.Tensor] = []
            z_by_stream: list[torch.Tensor] = []
            v_by_stream: list[torch.Tensor] = []

            for target in range(n_streams):
                strong = graph_mask[:, :, target, :].clone()
                strong[..., target] = False
                other = stream_index.view(1, 1, -1) != int(target)
                other = other.expand(batch_size, n_data, -1)
                weak = other & (~strong)

                cancelled_mean = torch.where(
                    strong, symbol_mean, torch.zeros_like(symbol_mean)
                )
                residual = y_data - torch.einsum(
                    "bdrn,bdn->bdr", h_data, cancelled_mean
                )

                # Strong neighbours retain only their residual uncertainty.
                # Weak neighbours are not cancelled and retain unit symbol power.
                interference_weight = torch.where(
                    strong,
                    symbol_var,
                    torch.full_like(symbol_var, float(constellation_energy.item())),
                )
                interference_weight = torch.where(
                    other,
                    interference_weight,
                    torch.zeros_like(interference_weight),
                )
                spatial_covariance = torch.einsum(
                    "bdrn,bdn,bdsn->bdrs",
                    h_data,
                    interference_weight,
                    h_data.conj(),
                )

                (
                    other_channel_error,
                    target_channel_variance,
                    target_channel_cross,
                ) = _channel_error_terms(
                    covariance,
                    symbol_mean,
                    symbol_var,
                    strong,
                    target,
                    covariance_mode=mode,
                )
                # The filter uses the unit-energy target cavity prior. The
                # scalar QAM likelihood below restores the exact |a|^2 target
                # variance, and the full-covariance mode also restores the
                # candidate-dependent cross term.
                scalar_noise = (
                    noise
                    + other_channel_error
                    + constellation_energy * target_channel_variance
                ).clamp_min(self.variance_floor)
                total_covariance = spatial_covariance + scalar_noise[
                    ..., None, None
                ].to(h_data.dtype) * identity
                total_covariance = 0.5 * (
                    total_covariance
                    + total_covariance.conj().transpose(-1, -2)
                )
                trace_scale = (
                    torch.diagonal(total_covariance, dim1=-2, dim2=-1)
                    .real.mean(dim=-1)
                    .clamp_min(1.0)
                )
                total_covariance = total_covariance + (
                    self.jitter * trace_scale
                )[..., None, None].to(h_data.dtype) * identity
                hermitian_error = torch.max(
                    torch.abs(
                        total_covariance
                        - total_covariance.conj().transpose(-1, -2)
                    )
                )
                max_hermitian = torch.maximum(max_hermitian, hermitian_error)

                h_target = h_data[..., target]
                filter_vector = torch.linalg.solve(
                    total_covariance, h_target.unsqueeze(-1)
                ).squeeze(-1)
                gain = torch.sum(h_target.conj() * filter_vector, dim=-1).real
                gain = gain.clamp_min(self.variance_floor)
                min_gain = torch.minimum(min_gain, gain.min())
                max_gain = torch.maximum(max_gain, gain.max())
                extrinsic_mean = (
                    torch.sum(filter_vector.conj() * residual, dim=-1) / gain
                )
                filter_norm_sq = torch.sum(
                    torch.abs(filter_vector) ** 2, dim=-1
                ).real
                gain_sq = gain.square().clamp_min(self.variance_floor)
                # Remove the unit-energy target channel-error contribution from
                # 1/g, then add its candidate-conditioned value.
                target_variance_coefficient = (
                    target_channel_variance * filter_norm_sq / gain_sq
                ).real.clamp_min(0.0)
                target_cross_coefficient = (
                    target_channel_cross
                    * filter_norm_sq.to(target_channel_cross.dtype)
                    / gain_sq.to(target_channel_cross.dtype)
                )
                base_extrinsic_var = (
                    1.0 / gain
                    - constellation_energy * target_variance_coefficient
                ).real.clamp_min(self.variance_floor)
                extrinsic_var = (
                    base_extrinsic_var
                    + constellation_energy * target_variance_coefficient
                ).real.clamp(self.variance_floor, self.variance_ceiling)

                candidate_variance = (
                    base_extrinsic_var[..., None]
                    + torch.abs(constellation).square().view(1, 1, -1)
                    * target_variance_coefficient[..., None]
                    + 2.0
                    * torch.real(
                        constellation.view(1, 1, -1)
                        * target_cross_coefficient[..., None]
                    )
                ).real.clamp(self.variance_floor, self.variance_ceiling)
                min_candidate_variance = torch.minimum(
                    min_candidate_variance, candidate_variance.min()
                )
                max_candidate_variance = torch.maximum(
                    max_candidate_variance, candidate_variance.max()
                )
                delta = extrinsic_mean[..., None] - constellation.view(1, 1, -1)
                symbol_logits = (
                    -torch.abs(delta) ** 2 / candidate_variance
                    - torch.log(candidate_variance)
                )
                logits_by_stream.append(symbol_logits)
                z_by_stream.append(extrinsic_mean)
                v_by_stream.append(extrinsic_var)

            # [B,N,D,M] for compatibility with the existing bridge/decoder.
            last_logits = torch.stack(logits_by_stream, dim=1)
            last_z = torch.stack(z_by_stream, dim=2)
            last_v = torch.stack(v_by_stream, dim=2)
            probabilities = torch.softmax(last_logits, dim=-1)
            proposed_mean, proposed_var = symbol_probs_to_mean_var(
                probabilities, constellation
            )
            proposed_mean = proposed_mean.permute(0, 2, 1).contiguous()
            proposed_var = proposed_var.permute(0, 2, 1).contiguous()

            old_second = symbol_var + torch.abs(symbol_mean) ** 2
            new_second = proposed_var + torch.abs(proposed_mean) ** 2
            damped_mean = (1.0 - beta) * symbol_mean + beta * proposed_mean
            damped_second = (1.0 - beta) * old_second + beta * new_second
            symbol_mean = damped_mean
            symbol_var = (
                damped_second - torch.abs(damped_mean) ** 2
            ).real.clamp(self.variance_floor, self.variance_ceiling)

        if last_logits is None:
            raise RuntimeError("Detector completed without symbol logits")
        bit_logits = symbol_logits_to_bit_logits(last_logits, bit_table)
        finite = bool(
            torch.isfinite(bit_logits).all().item()
            and torch.isfinite(symbol_mean).all().item()
            and torch.isfinite(symbol_var).all().item()
            and torch.isfinite(last_z).all().item()
            and torch.isfinite(last_v).all().item()
        )
        diagnostics = LMMSEEPDiagnostics(
            min_extrinsic_variance=float(last_v.detach().min().item()),
            max_extrinsic_variance=float(last_v.detach().max().item()),
            min_candidate_variance=float(min_candidate_variance.detach().item()),
            max_candidate_variance=float(max_candidate_variance.detach().item()),
            min_filter_gain=float(min_gain.detach().item()),
            max_filter_gain=float(max_gain.detach().item()),
            min_belief_variance=float(symbol_var.detach().min().item()),
            max_belief_variance=float(symbol_var.detach().max().item()),
            max_hermitian_error=float(max_hermitian.detach().item()),
            finite=finite,
        )
        return {
            "bit_logits": bit_logits,
            "symbol_logits": last_logits,
            "x_mean": symbol_mean.permute(0, 2, 1).contiguous(),
            "x_var": symbol_var.permute(0, 2, 1).contiguous(),
            "extrinsic_mean": last_z.permute(0, 2, 1).contiguous(),
            "extrinsic_var": last_v.permute(0, 2, 1).contiguous(),
            "graph_mask": graph_mask,
            "edge_density": (
                graph_mask.float().sum()
                / max(
                    float(batch_size * n_data * n_streams * max(n_streams - 1, 1)),
                    1.0,
                )
            ),
            "diagnostics": diagnostics.__dict__,
            "detector_version": DETECTOR_VERSION,
            "detector_iterations": iterations,
            "damping": beta,
            "covariance_mode": mode,
        }


def mathematical_self_test(device: torch.device | str = "cpu") -> dict[str, Any]:
    """Deterministic shape, scalar-equivalence, and equivariance checks."""
    dev = torch.device(device)
    torch.manual_seed(7411)

    # Single-stream scalar-equivalence test.
    batch, n_data, n_rx, n_streams = 2, 7, 3, 1
    h = (
        torch.randn(batch, n_streams, n_rx, n_data, device=dev)
        + 1j * torch.randn(batch, n_streams, n_rx, n_data, device=dev)
    ) / 2**0.5
    y = (
        torch.randn(batch, n_rx, n_data, device=dev)
        + 1j * torch.randn(batch, n_rx, n_data, device=dev)
    ) / 2**0.5
    noise = torch.tensor(0.27, device=dev)
    covariance = torch.zeros(
        n_streams, n_streams, n_data, dtype=torch.complex64, device=dev
    )
    graph = torch.zeros(
        batch, n_data, n_streams, n_streams, dtype=torch.bool, device=dev
    )
    detector = DampedExtrinsicLMMSEDetector(
        2, n_iter=1, damping=0.5, covariance_mode="none"
    ).to(dev)
    result = detector(
        y, h, covariance, torch.arange(n_data, device=dev), noise, graph
    )
    h0 = h[:, 0].permute(0, 2, 1)
    y0 = y.permute(0, 2, 1)
    norm = torch.sum(torch.abs(h0) ** 2, dim=-1).clamp_min(1e-8)
    expected_z = torch.sum(h0.conj() * y0, dim=-1) / norm
    expected_v = noise / norm
    scalar_mean_error = float(
        torch.max(torch.abs(result["extrinsic_mean"][:, 0] - expected_z)).item()
    )
    scalar_var_error = float(
        torch.max(torch.abs(result["extrinsic_var"][:, 0] - expected_v)).item()
    )

    # Multi-stream one-step equivalence to a manual unbiased spatial LMMSE
    # filter. This is the key repair relative to the scalar soft-PIC front end.
    batch, n_data, n_rx, n_streams = 2, 6, 4, 3
    h_multi = (
        torch.randn(batch, n_streams, n_rx, n_data, device=dev)
        + 1j * torch.randn(batch, n_streams, n_rx, n_data, device=dev)
    ) / 2**0.5
    x_multi = (
        torch.randn(batch, n_streams, n_data, device=dev)
        + 1j * torch.randn(batch, n_streams, n_data, device=dev)
    ) / 2**0.5
    y_multi = torch.einsum("bnrd,bnd->brd", h_multi, x_multi)
    noise_multi = torch.tensor(0.31, device=dev)
    cov_multi = torch.zeros(
        n_streams, n_streams, n_data, dtype=torch.complex64, device=dev
    )
    graph_multi = full_directed_graph(
        batch, n_data, n_streams, device=dev
    )
    one_step = DampedExtrinsicLMMSEDetector(
        2, n_iter=1, damping=0.5, covariance_mode="none"
    ).to(dev)(
        y_multi, h_multi, cov_multi, torch.arange(n_data, device=dev),
        noise_multi, graph_multi
    )
    h_bd = h_multi.permute(0, 3, 2, 1).contiguous()
    y_bd = y_multi.permute(0, 2, 1).contiguous()
    manual_z = []
    manual_v = []
    eye = torch.eye(n_rx, dtype=torch.complex64, device=dev).view(1, 1, n_rx, n_rx)
    for target in range(n_streams):
        mask = torch.arange(n_streams, device=dev) != target
        h_other = h_bd[..., mask]
        r_int = torch.einsum("bdrn,bdsn->bdrs", h_other, h_other.conj())
        r_int = r_int + noise_multi.to(torch.complex64) * eye
        h_t = h_bd[..., target]
        w = torch.linalg.solve(r_int, h_t.unsqueeze(-1)).squeeze(-1)
        gain = torch.sum(h_t.conj() * w, dim=-1).real.clamp_min(1e-8)
        manual_z.append(torch.sum(w.conj() * y_bd, dim=-1) / gain)
        manual_v.append(1.0 / gain)
    manual_z = torch.stack(manual_z, dim=1)
    manual_v = torch.stack(manual_v, dim=1)
    multi_mean_error = float(
        torch.max(torch.abs(one_step["extrinsic_mean"] - manual_z)).item()
    )
    multi_var_error = float(
        torch.max(torch.abs(one_step["extrinsic_var"] - manual_v)).item()
    )

    # Layer-permutation equivariance with a full graph.
    batch, n_data, n_rx, n_streams = 2, 5, 4, 3
    h = (
        torch.randn(batch, n_streams, n_rx, n_data, device=dev)
        + 1j * torch.randn(batch, n_streams, n_rx, n_data, device=dev)
    ) / 2**0.5
    x = (
        torch.randn(batch, n_streams, n_data, device=dev)
        + 1j * torch.randn(batch, n_streams, n_data, device=dev)
    ) / 2**0.5
    y = torch.einsum("bnrd,bnd->brd", h, x)
    covariance = torch.zeros(
        n_streams, n_streams, n_data, dtype=torch.complex64, device=dev
    )
    graph = full_directed_graph(batch, n_data, n_streams, device=dev)
    detector = DampedExtrinsicLMMSEDetector(
        2, n_iter=3, damping=0.5, covariance_mode="none"
    ).to(dev)
    reference = detector(
        y, h, covariance, torch.arange(n_data, device=dev), 0.4, graph
    )
    permutation = torch.tensor([2, 0, 1], device=dev)
    inverse = torch.argsort(permutation)
    h_perm = h[:, permutation]
    graph_perm = graph[:, :, permutation][:, :, :, permutation]
    permuted = detector(
        y, h_perm, covariance[permutation][:, permutation],
        torch.arange(n_data, device=dev), 0.4, graph_perm
    )
    equivariance_error = float(
        torch.max(
            torch.abs(
                reference["bit_logits"]
                - permuted["bit_logits"][:, inverse]
            )
        ).item()
    )

    checks = {
        "single_stream_extrinsic_mean": scalar_mean_error < 2e-4,
        "single_stream_extrinsic_variance": scalar_var_error < 2e-4,
        "multi_stream_spatial_lmmse_mean": multi_mean_error < 3e-4,
        "multi_stream_spatial_lmmse_variance": multi_var_error < 3e-4,
        "layer_permutation_equivariance": equivariance_error < 3e-4,
        "finite": bool(reference["diagnostics"]["finite"]),
        "positive_extrinsic_variance": bool(
            reference["extrinsic_var"].min().item() > 0.0
        ),
        "zero_self_edges": bool(
            not torch.diagonal(reference["graph_mask"], dim1=-2, dim2=-1)
            .any()
            .item()
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "scalar_mean_max_abs_error": scalar_mean_error,
        "scalar_variance_max_abs_error": scalar_var_error,
        "multi_stream_mean_max_abs_error": multi_mean_error,
        "multi_stream_variance_max_abs_error": multi_var_error,
        "permutation_max_abs_error": equivariance_error,
        "detector_version": DETECTOR_VERSION,
        "device": str(dev),
    }
