from __future__ import annotations

"""Compatibility support for Sionna 2.0.1 K-best on clusters without Triton.

Sionna 2.0.1 decorates ``List2LLRSimple._fused_equal_any`` with
``torch.compile(fullgraph=True)``. The compiled function evaluates exactly

    (path_inds == c).any(dim=-2)

and exists only to fuse the comparison and reduction. Some Alliance PyTorch
builds provide CUDA execution but do not install Triton, so the compiled helper
raises ``torch._inductor.exc.TritonMissing``.

Gate-1 uses the mathematically identical eager expression. This changes only
the implementation backend of the list-to-LLR helper, not K-best candidates,
distance metrics, LLR equations, or decoded results.
"""

from typing import Any
import importlib.util

import torch


K_BEST_COMPAT_VERSION = "sionna_2_0_1_list2llr_eager_exact_v1"


def _eager_equal_any(path_inds: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Exact eager form of Sionna 2.0.1's compiled helper."""
    return (path_inds == c).any(dim=-2)


setattr(_eager_equal_any, "_bayesroute_kbest_eager_exact", True)


def _triton_importable() -> bool:
    try:
        return importlib.util.find_spec("triton") is not None
    except Exception:
        return False


def _semantic_self_test() -> dict[str, Any]:
    # Shapes follow Sionna after two singleton dimensions are inserted:
    # path indices [B, paths, streams, 1, 1]
    # reference indices [1, 1, 1, constellation/2, bits]
    path_inds = torch.tensor(
        [[[[[0]], [[1]]], [[[2]], [[3]]]]],
        dtype=torch.int32,
    )
    references = torch.tensor(
        [[[[[0, 1], [2, 3]]]]],
        dtype=torch.int32,
    )
    expected = (path_inds == references).any(dim=-2)
    observed = _eager_equal_any(path_inds, references)
    return {
        "passed": bool(torch.equal(observed, expected)),
        "expected_shape": list(expected.shape),
        "observed_shape": list(observed.shape),
        "semantics": "(path_inds == c).any(dim=-2)",
    }


def configure_sionna_kbest_compat(*, force_eager: bool = True) -> dict[str, Any]:
    """Configure and verify the exact eager List2LLR helper.

    The patch is process-local and does not edit the installed Sionna package.
    """
    import sionna
    from sionna.phy.mimo.utils import List2LLRSimple

    current = List2LLRSimple._fused_equal_any
    already_patched = bool(
        getattr(current, "_bayesroute_kbest_eager_exact", False)
    )

    if force_eager and not already_patched:
        List2LLRSimple._fused_equal_any = staticmethod(_eager_equal_any)

    active = List2LLRSimple._fused_equal_any
    backend = (
        "eager_exact"
        if getattr(active, "_bayesroute_kbest_eager_exact", False)
        else "sionna_torch_compile"
    )

    semantic = _semantic_self_test()

    # Test the active class method as it will be called by a detector instance.
    path_inds = torch.tensor(
        [[[[[0]], [[1]]], [[[2]], [[3]]]]],
        dtype=torch.int32,
    )
    references = torch.tensor(
        [[[[[0, 1], [2, 3]]]]],
        dtype=torch.int32,
    )
    expected = (path_inds == references).any(dim=-2)
    observed = List2LLRSimple._fused_equal_any(path_inds, references)
    active_exact = bool(torch.equal(observed, expected))

    return {
        "passed": bool(
            semantic["passed"]
            and active_exact
            and (not force_eager or backend == "eager_exact")
        ),
        "compat_version": K_BEST_COMPAT_VERSION,
        "backend": backend,
        "force_eager": bool(force_eager),
        "already_patched": already_patched,
        "active_semantics_exact": active_exact,
        "semantic_self_test": semantic,
        "sionna_version": str(getattr(sionna, "__version__", "unknown")),
        "torch_version": str(torch.__version__),
        "triton_importable": _triton_importable(),
        "installed_sionna_files_modified": False,
    }
