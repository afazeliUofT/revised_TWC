from __future__ import annotations
import importlib
import traceback
import torch


def check_sionna(device: torch.device | str | None = None, bits_per_symbol: int = 4) -> dict:
    """Execute a real Sionna Mapper/Demapper round trip and gradient check."""
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    rep = {
        "sionna_import": False,
        "version": None,
        "mapper_constructed": False,
        "demapper_constructed": False,
        "roundtrip_bit_errors": None,
        "gradient_finite_nonzero": False,
        "error": None,
    }
    try:
        sionna = importlib.import_module("sionna")
        rep["sionna_import"] = True
        rep["version"] = getattr(sionna, "__version__", "unknown")
        mapping = importlib.import_module("sionna.phy.mapping")
        Mapper = getattr(mapping, "Mapper")
        Demapper = getattr(mapping, "Demapper")
        Constellation = getattr(mapping, "Constellation")

        mapper = Mapper("qam", int(bits_per_symbol), device=str(dev))
        demapper = Demapper("app", "qam", int(bits_per_symbol), device=str(dev))
        constellation = Constellation("qam", int(bits_per_symbol), device=str(dev))
        rep["mapper_constructed"] = True
        rep["demapper_constructed"] = True
        rep["constellation_power"] = float(torch.mean(torch.abs(constellation()) ** 2).item())

        n_bits = 256
        bit_index = torch.arange(n_bits, device=dev)
        bits = ((bit_index * 13 + bit_index // 7) % 2).to(torch.float32).view(4, -1)
        # Mapper input length must be a multiple of Q.
        keep = (bits.shape[-1] // int(bits_per_symbol)) * int(bits_per_symbol)
        bits = bits[:, :keep]
        symbols = mapper(bits)
        y = symbols.detach().clone().requires_grad_(True)
        no = torch.tensor(1e-4, dtype=torch.float32, device=dev)
        llr = demapper(y, no=no)
        hard = (llr >= 0).to(bits.dtype)
        rep["mapped_shape"] = list(symbols.shape)
        rep["llr_shape"] = list(llr.shape)
        rep["roundtrip_bit_errors"] = int(torch.sum(hard != bits).item())
        objective = torch.mean(llr.square())
        grad = torch.autograd.grad(objective, y, retain_graph=False, create_graph=False)[0]
        rep["gradient_norm"] = float(torch.linalg.vector_norm(torch.view_as_real(grad)).item())
        rep["gradient_finite_nonzero"] = bool(
            torch.isfinite(grad).all().item() and rep["gradient_norm"] > 0.0
        )
    except Exception as exc:  # pragma: no cover - exercised on Rorqual
        rep["error"] = repr(exc) + "\n" + traceback.format_exc(limit=4)

    rep["passed"] = bool(
        rep["sionna_import"]
        and rep["mapper_constructed"]
        and rep["demapper_constructed"]
        and rep["roundtrip_bit_errors"] == 0
        and rep["gradient_finite_nonzero"]
    )
    return rep
