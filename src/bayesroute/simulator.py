from __future__ import annotations
from dataclasses import dataclass
import torch
from .pilots import make_grid, pilot_indices, data_indices, make_orthogonal_dmrs, port_metadata
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

class UplinkToySimulator:
    """Compact MU-MIMO OFDM simulator for BayesRoute principle testing."""

    def __init__(self, cfg, device: torch.device):
        self.cfg = cfg
        self.device = device
        sys = cfg.system
        self.n_rx = int(sys.n_rx)
        self.n_layers = int(sys.n_layers)
        self.n_subcarriers = int(sys.n_subcarriers)
        self.n_symbols = int(sys.n_symbols)
        self.dmrs_symbols = [int(x) for x in sys.dmrs_symbols]
        self.bits_per_symbol = int(sys.bits_per_symbol)
        self.coords = make_grid(self.n_subcarriers, self.n_symbols, device=device)
        self.pilot_idx = pilot_indices(self.n_subcarriers, self.n_symbols, self.dmrs_symbols, device=device)
        self.data_idx = data_indices(self.n_subcarriers, self.n_symbols, self.dmrs_symbols, device=device)
        self.phi = make_orthogonal_dmrs(self.n_layers, int(self.pilot_idx.numel()), device=device)
        self.port_meta = port_metadata(self.n_layers)

    @property
    def n_resource_elements(self) -> int:
        return int(self.n_subcarriers * self.n_symbols)

    def sample(self, batch_size: int, snr_db: float | None = None) -> Batch:
        sys = self.cfg.system
        if snr_db is None:
            snr_min = float(sys.snr_db_min)
            snr_max = float(sys.snr_db_max)
            snr_db_t = snr_min + (snr_max - snr_min) * torch.rand((), device=self.device)
            snr_db = float(snr_db_t.item())
        no = torch.tensor(10.0 ** (-float(snr_db) / 10.0), dtype=torch.float32, device=self.device)
        bps = self.bits_per_symbol
        d = int(self.data_idx.numel())
        bits = torch.randint(0, 2, (batch_size, self.n_layers, d, bps), device=self.device)
        data_syms = bits_to_symbols(bits, bps)
        x = torch.zeros((batch_size, self.n_layers, self.n_resource_elements), dtype=torch.complex64, device=self.device)
        x[:, :, self.data_idx] = data_syms
        x[:, :, self.pilot_idx] = self.phi.view(1, self.n_layers, -1)
        h = generate_low_rank_channel(
            batch_size=batch_size,
            n_layers=self.n_layers,
            n_rx=self.n_rx,
            coords=self.coords,
            true_rank=int(sys.channel_true_rank),
            length_f=float(sys.channel_length_f),
            length_t=float(sys.channel_length_t),
            seed=int(self.cfg.seed) + 17,
        )
        clean = torch.sum(h * x[:, :, None, :], dim=1)
        noise = complex_normal(clean.shape, device=self.device, scale=torch.sqrt(no).item())
        y = clean + noise
        return Batch(y=y, h=h, x=x, data_bits=bits.float(), data_symbols=data_syms,
                     coords=self.coords, pilot_idx=self.pilot_idx, data_idx=self.data_idx,
                     phi=self.phi, noise_var=no, snr_db=float(snr_db), port_meta=self.port_meta)
