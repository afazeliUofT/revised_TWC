from __future__ import annotations
import importlib
import traceback
import torch
from .config import canonical_torch_device


def check_sionna(device: torch.device | str | None = None, bits_per_symbol: int = 4) -> dict:
    """Execute a real Sionna Mapper/Demapper round trip and gradient check."""
    requested = device if device is not None else (
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )
    dev = canonical_torch_device(requested)
    device_string = str(dev)
    rep = {
        "sionna_import": False,
        "version": None,
        "requested_device": str(requested),
        "normalized_device": device_string,
        "available_devices": None,
        "device_available": False,
        "device_match": False,
        "mapper_constructed": False,
        "demapper_constructed": False,
        "roundtrip_bit_errors": None,
        "internal_mapper_max_abs_error": None,
        "internal_mapper_matches_sionna": False,
        "gradient_finite_nonzero": False,
        "error": None,
    }
    try:
        sionna = importlib.import_module("sionna")
        rep["sionna_import"] = True
        rep["version"] = getattr(sionna, "__version__", "unknown")
        phy = importlib.import_module("sionna.phy")
        sionna_config = getattr(phy, "config")
        rep["available_devices"] = list(sionna_config.available_devices)
        rep["device_available"] = device_string in rep["available_devices"]
        if not rep["device_available"]:
            raise RuntimeError(
                f"Sionna device {device_string!r} is unavailable; "
                f"available={rep['available_devices']}"
            )

        mapping = importlib.import_module("sionna.phy.mapping")
        Mapper = getattr(mapping, "Mapper")
        Demapper = getattr(mapping, "Demapper")
        Constellation = getattr(mapping, "Constellation")

        mapper = Mapper("qam", int(bits_per_symbol), device=device_string)
        demapper = Demapper("app", "qam", int(bits_per_symbol), device=device_string)
        constellation = Constellation("qam", int(bits_per_symbol), device=device_string)
        rep["mapper_constructed"] = True
        rep["demapper_constructed"] = True
        rep["constellation_power"] = float(
            torch.mean(torch.abs(constellation()) ** 2).item()
        )

        # Exhaustively enumerate every binary label once. This verifies the full
        # point ordering, not only a sampled subset of constellation points.
        q = int(bits_per_symbol)
        n_points = 2 ** q
        indices = torch.arange(n_points, device=dev, dtype=torch.long)
        shifts = torch.arange(q - 1, -1, -1, device=dev, dtype=torch.long)
        all_grouped_bits = ((indices[:, None] >> shifts[None, :]) & 1).to(
            torch.float32
        )
        all_bits = all_grouped_bits.reshape(1, -1)
        all_symbols = mapper(all_bits)

        from .qam import bits_to_symbols
        internal_all_symbols = bits_to_symbols(
            all_grouped_bits.view(1, n_points, q), q
        )
        mapping_error = torch.max(
            torch.abs(internal_all_symbols - all_symbols)
        ).item()
        rep["enumerated_constellation_points"] = n_points
        rep["internal_mapper_max_abs_error"] = float(mapping_error)
        rep["internal_mapper_matches_sionna"] = bool(mapping_error < 1e-7)

        # Repeat the complete constellation four times to exercise batching,
        # demapping, hard decisions, and gradients.
        bits = all_bits.repeat(4, 1)
        symbols = mapper(bits)
        y = symbols.detach().clone().requires_grad_(True)
        no = torch.tensor(1e-2, dtype=torch.float32, device=dev)
        llr = demapper(y, no=no)
        hard = (llr >= 0).to(bits.dtype)
        rep["mapped_shape"] = list(symbols.shape)
        rep["llr_shape"] = list(llr.shape)
        rep["symbols_device"] = str(symbols.device)
        rep["llr_device"] = str(llr.device)
        rep["roundtrip_bit_errors"] = int(torch.sum(hard != bits).item())

        objective = torch.mean(llr.square())
        grad = torch.autograd.grad(
            objective, y, retain_graph=False, create_graph=False
        )[0]
        rep["gradient_device"] = str(grad.device)
        rep["gradient_norm"] = float(
            torch.linalg.vector_norm(torch.view_as_real(grad)).item()
        )
        rep["gradient_finite_nonzero"] = bool(
            torch.isfinite(grad).all().item() and rep["gradient_norm"] > 0.0
        )
        rep["device_match"] = bool(
            symbols.device == dev and llr.device == dev and grad.device == dev
        )
    except Exception as exc:  # pragma: no cover - exercised on Rorqual
        rep["error"] = repr(exc) + "\n" + traceback.format_exc(limit=4)

    rep["passed"] = bool(
        rep["sionna_import"]
        and rep["device_available"]
        and rep["device_match"]
        and rep["mapper_constructed"]
        and rep["demapper_constructed"]
        and rep["roundtrip_bit_errors"] == 0
        and rep["internal_mapper_matches_sionna"]
        and rep["gradient_finite_nonzero"]
    )
    return rep
