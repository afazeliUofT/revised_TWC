from __future__ import annotations
from dataclasses import dataclass
import torch
from .pilots import (
    PILOT_MODEL_SCOPE,
    make_grid,
    pilot_indices,
    data_indices,
    make_orthogonal_dmrs,
    port_metadata,
)
from .qam import bits_to_symbols
from .channels import generate_low_rank_channel, complex_normal


@dataclass
class Batch:
    y: torch.Tensor                  # [B, RX, R]
    h: torch.Tensor                  # [B, N, RX, R]
    x: torch.Tensor                  # [B, N, R]
    data_bits: torch.Tensor          # [B, N, D, Q]
    data_symbols: torch.Tensor       # [B, N, D]
    coords: torch.Tensor             # [R, 2]
    pilot_idx: torch.Tensor          # [P]
    data_idx: torch.Tensor           # [D]
    phi: torch.Tensor                # [N, P]
    noise_var: torch.Tensor          # scalar tensor
    snr_db: float
    port_meta: list[dict]
    pilot_model_scope: str


class UplinkToySimulator:
    """Gate-0 MU-MIMO OFDM simulator for testing the receiver principle.

    The data constellation is taken from Sionna. The pilot codebook is an
    explicit orthogonal abstraction; it is not the final 3GPP PUSCH grid.
    """

    def __init__(self, cfg, device: torch.device):
        self.cfg = cfg
        self.device = device
        sys_cfg = cfg.system
        self.n_rx = int(sys_cfg.n_rx)
        self.n_layers = int(sys_cfg.n_layers)
        self.n_subcarriers = int(sys_cfg.n_subcarriers)
        self.n_symbols = int(sys_cfg.n_symbols)
        self.dmrs_symbols = [int(x) for x in sys_cfg.dmrs_symbols]
        self.bits_per_symbol = int(sys_cfg.bits_per_symbol)
        self.channel_seed = int(sys_cfg.get("channel_seed", int(cfg.seed) + 17))
        self.coords = make_grid(self.n_subcarriers, self.n_symbols, device=device)
        self.pilot_idx = pilot_indices(
            self.n_subcarriers, self.n_symbols, self.dmrs_symbols, device=device
        )
        self.data_idx = data_indices(
            self.n_subcarriers, self.n_symbols, self.dmrs_symbols, device=device
        )
        self.phi = make_orthogonal_dmrs(
            self.n_layers, int(self.pilot_idx.numel()), device=device
        )
        self.port_meta = port_metadata(self.n_layers)
        self.pilot_model_scope = PILOT_MODEL_SCOPE

    @property
    def n_resource_elements(self) -> int:
        return int(self.n_subcarriers * self.n_symbols)

    def sample(self, batch_size: int, snr_db: float | None = None) -> Batch:
        sys_cfg = self.cfg.system
        if snr_db is None:
            snr_min = float(sys_cfg.snr_db_min)
            snr_max = float(sys_cfg.snr_db_max)
            snr_db_t = snr_min + (snr_max - snr_min) * torch.rand((), device=self.device)
            snr_db = float(snr_db_t.item())
        noise_var = torch.tensor(
            10.0 ** (-float(snr_db) / 10.0),
            dtype=torch.float32,
            device=self.device,
        )
        bps = self.bits_per_symbol
        d = int(self.data_idx.numel())
        bits = torch.randint(
            0, 2, (batch_size, self.n_layers, d, bps), device=self.device
        )
        # Uses Sionna's constellation point ordering through qam.bits_to_symbols.
        data_syms = bits_to_symbols(bits, bps)
        x = torch.zeros(
            (batch_size, self.n_layers, self.n_resource_elements),
            dtype=torch.complex64,
            device=self.device,
        )
        x[:, :, self.data_idx] = data_syms
        x[:, :, self.pilot_idx] = self.phi.view(1, self.n_layers, -1)
        h = generate_low_rank_channel(
            batch_size=batch_size,
            n_layers=self.n_layers,
            n_rx=self.n_rx,
            coords=self.coords,
            true_rank=int(sys_cfg.channel_true_rank),
            length_f=float(sys_cfg.channel_length_f),
            length_t=float(sys_cfg.channel_length_t),
            seed=self.channel_seed,
            layer_power_spread_db=float(sys_cfg.get("layer_power_spread_db", 0.0)),
        )
        clean = torch.sum(h * x[:, :, None, :], dim=1)
        noise = complex_normal(
            clean.shape, device=self.device, scale=torch.sqrt(noise_var).item()
        )
        y = clean + noise
        return Batch(
            y=y,
            h=h,
            x=x,
            data_bits=bits.float(),
            data_symbols=data_syms,
            coords=self.coords,
            pilot_idx=self.pilot_idx,
            data_idx=self.data_idx,
            phi=self.phi,
            noise_var=noise_var,
            snr_db=float(snr_db),
            port_meta=self.port_meta,
            pilot_model_scope=self.pilot_model_scope,
        )
