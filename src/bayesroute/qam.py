from __future__ import annotations
from functools import lru_cache
import torch
from .config import canonical_torch_device


def _device_string(device: torch.device | str | None) -> str:
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    return str(canonical_torch_device(device))


@lru_cache(maxsize=16)
def _cached_sionna_points(bits_per_symbol: int, device_string: str) -> torch.Tensor:
    """Return the Sionna QAM points in Sionna's binary-label order."""
    try:
        from sionna.phy.mapping import Constellation
    except Exception as exc:  # pragma: no cover - exercised on Rorqual
        raise RuntimeError(
            "Sionna PHY is required. Run scripts/rorqual_setup_env.py before the smoke test."
        ) from exc
    const = Constellation("qam", int(bits_per_symbol), device=device_string)
    points = const()
    return points.to(torch.device(device_string), dtype=torch.complex64).detach()


def make_qam_constellation(bits_per_symbol: int, device: torch.device | None = None):
    """Return Sionna's unit-power square-QAM constellation and binary labels.

    Sionna orders constellation points by the natural binary value of the bit
    label. Keeping this order makes the analytical detector and Sionna Mapper
    use the same labeling.
    """
    if int(bits_per_symbol) % 2 != 0:
        raise ValueError("Only square QAM with an even number of bits per symbol is supported.")
    device_string = _device_string(device)
    const = _cached_sionna_points(int(bits_per_symbol), device_string)
    m = 2 ** int(bits_per_symbol)
    labels = []
    for idx in range(m):
        labels.append([
            (idx >> (int(bits_per_symbol) - 1 - k)) & 1
            for k in range(int(bits_per_symbol))
        ])
    bit_table = torch.tensor(labels, dtype=torch.float32, device=const.device)
    return const, bit_table


def bits_to_symbols(bits: torch.Tensor, bits_per_symbol: int) -> torch.Tensor:
    """Map [..., number_of_symbols, Q] bits with Sionna's QAM labeling."""
    if bits.shape[-1] != int(bits_per_symbol):
        raise ValueError(
            f"Expected last dimension {bits_per_symbol}, received {bits.shape[-1]}."
        )
    const, _ = make_qam_constellation(int(bits_per_symbol), bits.device)
    powers = 2 ** torch.arange(
        int(bits_per_symbol) - 1, -1, -1, device=bits.device, dtype=torch.long
    )
    idx = torch.sum(bits.long() * powers, dim=-1)
    return const[idx]


def symbol_logits_to_bit_logits(symbol_logits: torch.Tensor, bit_table: torch.Tensor) -> torch.Tensor:
    """Convert symbol logits [..., M] to LLRs log P(bit=1)/P(bit=0)."""
    logits = []
    for q in range(bit_table.shape[-1]):
        mask1 = bit_table[:, q] > 0.5
        mask0 = ~mask1
        l1 = torch.logsumexp(symbol_logits[..., mask1], dim=-1)
        l0 = torch.logsumexp(symbol_logits[..., mask0], dim=-1)
        logits.append(l1 - l0)
    return torch.stack(logits, dim=-1)


def symbol_probs_to_mean_var(probs: torch.Tensor, constellation: torch.Tensor):
    shape = *([1] * (probs.ndim - 1)), -1
    points = constellation.view(*shape)
    mean = torch.sum(probs * points, dim=-1)
    second = torch.sum(probs * (torch.abs(constellation) ** 2).view(*shape), dim=-1)
    var = torch.clamp(second - torch.abs(mean) ** 2, min=1e-7)
    return mean, var
