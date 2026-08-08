from __future__ import annotations
import torch


def _gray_to_binary_int(g: int) -> int:
    b = 0
    while g:
        b ^= g
        g >>= 1
    return b


def make_qam_constellation(bits_per_symbol: int, device: torch.device | None = None):
    """Create square Gray-labeled QAM constellation with unit average power.

    The returned order is natural binary bit order. This keeps mapping and
    demapping internally consistent and simple.
    """
    if bits_per_symbol % 2 != 0:
        raise ValueError("Only square QAM with an even number of bits per symbol is supported.")
    m_side = 2 ** (bits_per_symbol // 2)
    m = 2 ** bits_per_symbol
    symbols = []
    bit_table = []
    for idx in range(m):
        bits = [(idx >> (bits_per_symbol - 1 - k)) & 1 for k in range(bits_per_symbol)]
        i_gray = 0
        q_gray = 0
        for b in bits[: bits_per_symbol // 2]:
            i_gray = (i_gray << 1) | b
        for b in bits[bits_per_symbol // 2 :]:
            q_gray = (q_gray << 1) | b
        i_bin = _gray_to_binary_int(i_gray)
        q_bin = _gray_to_binary_int(q_gray)
        i_level = 2 * i_bin - (m_side - 1)
        q_level = 2 * q_bin - (m_side - 1)
        symbols.append(complex(i_level, q_level))
        bit_table.append(bits)
    const = torch.tensor(symbols, dtype=torch.complex64, device=device)
    const = const / torch.sqrt(torch.mean(torch.abs(const) ** 2))
    bit_table_t = torch.tensor(bit_table, dtype=torch.float32, device=device)
    return const, bit_table_t


def bits_to_symbols(bits: torch.Tensor, bits_per_symbol: int) -> torch.Tensor:
    """Map bits with shape [..., bits_per_symbol] to complex QAM symbols."""
    const, _ = make_qam_constellation(bits_per_symbol, bits.device)
    powers = 2 ** torch.arange(bits_per_symbol - 1, -1, -1, device=bits.device)
    idx = torch.sum(bits.long() * powers, dim=-1)
    return const[idx]


def symbol_logits_to_bit_logits(symbol_logits: torch.Tensor, bit_table: torch.Tensor) -> torch.Tensor:
    """Convert symbol logits [..., M] to bit logits [..., Q]."""
    logits = []
    for q in range(bit_table.shape[-1]):
        mask1 = bit_table[:, q] > 0.5
        mask0 = ~mask1
        l1 = torch.logsumexp(symbol_logits[..., mask1], dim=-1)
        l0 = torch.logsumexp(symbol_logits[..., mask0], dim=-1)
        logits.append(l1 - l0)
    return torch.stack(logits, dim=-1)


def symbol_probs_to_mean_var(probs: torch.Tensor, constellation: torch.Tensor):
    mean = torch.sum(probs * constellation.view(*([1] * (probs.ndim - 1)), -1), dim=-1)
    second = torch.sum(probs * (torch.abs(constellation) ** 2).view(*([1] * (probs.ndim - 1)), -1), dim=-1)
    var = torch.clamp(second - torch.abs(mean) ** 2, min=1e-7)
    return mean, var
