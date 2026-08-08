from __future__ import annotations
import importlib
import traceback


def check_sionna() -> dict:
    """Check Sionna installation and a minimal PHY mapping call.

    This package keeps the main receiver implementation in PyTorch for clarity.
    Sionna is still checked because final TWC experiments should remain compatible
    with Sionna-based link-level blocks.
    """
    report = {"sionna_import": False, "version": None, "phy_mapping_api": False, "error": None}
    try:
        sionna = importlib.import_module("sionna")
        report["sionna_import"] = True
        report["version"] = getattr(sionna, "__version__", "unknown")
        try:
            mapping = importlib.import_module("sionna.phy.mapping")
            # API existence check. We avoid making assumptions about all constructor details.
            report["Mapper_available"] = hasattr(mapping, "Mapper")
            report["Demapper_available"] = hasattr(mapping, "Demapper")
            report["Constellation_available"] = hasattr(mapping, "Constellation")
            report["phy_mapping_api"] = bool(report["Mapper_available"] and report["Demapper_available"])
        except Exception as api_exc:
            report["error"] = "Sionna import succeeded, mapping API check failed: " + repr(api_exc)
    except Exception as exc:
        report["error"] = repr(exc) + "\n" + traceback.format_exc(limit=2)
    report["passed"] = bool(report["sionna_import"] and report["phy_mapping_api"])
    return report
