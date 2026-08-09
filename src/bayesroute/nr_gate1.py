from __future__ import annotations

"""Gate-1 bridge between the Gate-0 BayesRoute receiver and Sionna NR PUSCH.

The module deliberately uses Sionna's PUSCHConfig/PUSCHTransmitter/PUSCHReceiver,
3GPP TR 38.901 channel models, and TBEncoder/TBDecoder.  It does not recreate
NR DMRS or LDPC processing with local approximations.
"""

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Iterable
import hashlib
import json
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .models import (
    BayesRouteDetector,
    LowRankPosteriorOperator,
    coupling_matrix,
    coupling_selection_mask,
    edge_density,
)

GATE1_NR_VERSION = "gate1_nr_integration_v1"


@dataclass(frozen=True)
class NRCase:
    name: str
    scenario: str
    num_users: int
    num_rx_ant: int
    num_layers_per_user: int = 1
    num_prb: int = 4
    subcarrier_spacing_khz: int = 30
    carrier_frequency_hz: float = 3.5e9
    dmrs_config_type: int = 1
    dmrs_length: int = 1
    dmrs_additional_position: int = 1
    num_cdm_groups_without_data: int = 2
    dmrs_ports: tuple[int, ...] = (0, 1)
    mcs_index: int = 10
    mcs_table: int = 1
    speed_mps: float = 3.0
    delay_spread_s: float = 100e-9
    n_cell_id: int = 42

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "NRCase":
        # Configuration entries may carry workflow-only keys such as
        # ``run_kbest`` or ``kbest_k``. Keep the scientific case contract
        # strict while safely ignoring those orchestration fields.
        allowed = {item.name for item in fields(cls)}
        data = {key: item for key, item in dict(value).items() if key in allowed}
        data["dmrs_ports"] = tuple(int(x) for x in data["dmrs_ports"])
        return cls(**data)

    @property
    def num_streams(self) -> int:
        return int(self.num_users * self.num_layers_per_user)

    def validate(self) -> None:
        scenario = self.scenario.lower()
        if scenario not in {"umi", "uma", "cdl-a", "cdl-c", "cdl-d", "awgn"}:
            raise ValueError(f"Unsupported Gate-1 scenario: {self.scenario}")
        if self.num_users <= 0:
            raise ValueError(f"{self.name}: num_users must be positive")
        if self.num_layers_per_user not in (1, 2, 3, 4):
            raise ValueError(
                f"{self.name}: num_layers_per_user must be in [1,2,3,4]"
            )
        if len(self.dmrs_ports) != self.num_streams:
            raise ValueError(
                f"{self.name}: expected num_users*num_layers_per_user="
                f"{self.num_streams} DMRS ports, received {self.dmrs_ports}"
            )
        if len(set(self.dmrs_ports)) != len(self.dmrs_ports):
            raise ValueError(f"{self.name}: DMRS ports must be unique")
        if self.dmrs_config_type not in (1, 2):
            raise ValueError("DMRS config type must be 1 or 2")
        if self.dmrs_length not in (1, 2):
            raise ValueError("DMRS length must be 1 or 2")
        if scenario.startswith("cdl-") and self.num_users != 1:
            raise ValueError("Sionna CDL Gate-1 cases must use one transmitter")


@dataclass
class NRGridDescription:
    coords: torch.Tensor
    pilot_idx: torch.Tensor
    data_idx: torch.Tensor
    phi: torch.Tensor
    pilot_grid: torch.Tensor
    pilot_mask: torch.Tensor
    reserved_mask: torch.Tensor
    reserved_zero_mask: torch.Tensor
    data_mask: torch.Tensor
    effective_subcarrier_ind: torch.Tensor
    port_metadata: list[dict[str, Any]]
    num_ofdm_symbols: int
    num_effective_subcarriers: int
    fft_size: int
    bits_per_symbol: int
    num_users: int
    num_layers_per_user: int

    @property
    def num_streams(self) -> int:
        return int(self.num_users * self.num_layers_per_user)

    @property
    def num_resource_elements(self) -> int:
        return int(self.num_ofdm_symbols * self.num_effective_subcarriers)

    @property
    def num_data_symbols(self) -> int:
        return int(self.data_idx.numel())

    @property
    def num_pilot_observations(self) -> int:
        return int(self.pilot_idx.numel())


@dataclass
class NRBatch:
    y: torch.Tensor                 # [B,RX,R]
    h: torch.Tensor                 # [B,N_stream,RX,R]
    information_bits: torch.Tensor  # [B,U,K]
    coded_bits: torch.Tensor        # [B,N_stream,D,Q]
    x_grid: torch.Tensor            # [B,N_stream,S,F_eff]
    noise_var: torch.Tensor
    ebno_db: float
    phi: torch.Tensor
    pilot_idx: torch.Tensor
    data_idx: torch.Tensor
    port_metadata: list[dict[str, Any]]
    raw_y: torch.Tensor
    raw_h: torch.Tensor


@dataclass
class NRContext:
    case: NRCase
    transmitter: Any
    pusch_configs: list[Any]
    grid: NRGridDescription
    channel_model: Any
    channel: Any
    device: torch.device

    def new_topology(self, batch_size: int) -> None:
        scenario = self.case.scenario.lower()
        if scenario not in {"umi", "uma"}:
            return
        from sionna.phy.channel import gen_single_sector_topology

        topology = gen_single_sector_topology(
            int(batch_size),
            int(self.case.num_users),
            scenario,
            min_ut_velocity=float(self.case.speed_mps),
            max_ut_velocity=float(self.case.speed_mps),
            device=str(self.device),
        )
        self.channel_model.set_topology(*topology)

    def sample(self, batch_size: int, ebno_db: float) -> NRBatch:
        from sionna.phy.utils import ebnodb2no

        self.new_topology(int(batch_size))
        x, b = self.transmitter(int(batch_size))
        coded = self.transmitter._tb_encoder(b)
        no = ebnodb2no(
            float(ebno_db),
            self.transmitter._num_bits_per_symbol,
            self.transmitter._target_coderate,
            self.transmitter.resource_grid,
            device=str(self.device),
        )
        if self.case.scenario.lower() == "awgn":
            # AWGN does not superpose users. This branch is used only for codec
            # checks and is not part of Gate-1 multiuser evidence.
            raise RuntimeError("Use codec_roundtrip() for AWGN-only checks")
        y, h = self.channel(x, no)
        return adapt_sionna_batch(
            self.transmitter,
            self.grid,
            x=x,
            b=b,
            coded=coded,
            y=y,
            h=h,
            no=no,
            ebno_db=float(ebno_db),
        )


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_device(device: torch.device | str) -> torch.device:
    dev = torch.device(device)
    if dev.type == "cuda" and dev.index is None:
        dev = torch.device(f"cuda:{torch.cuda.current_device()}")
    return dev


def _to_list(value: Any) -> list[Any]:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def build_pusch_configs(case: NRCase) -> list[Any]:
    """Build one standard PUSCH configuration per user.

    Sionna requires the same number of layers for every transmitter in one
    PUSCHTransmitter. Gate-1 therefore supports homogeneous layer counts across
    users and keeps an explicit contiguous user/layer/port flattening map.
    """
    from sionna.phy.nr import PUSCHConfig

    case.validate()
    configs: list[Any] = []
    layers = int(case.num_layers_per_user)
    for user_index in range(int(case.num_users)):
        start = user_index * layers
        ports = [int(x) for x in case.dmrs_ports[start:start + layers]]
        config = PUSCHConfig()
        config.carrier.subcarrier_spacing = int(case.subcarrier_spacing_khz)
        config.carrier.n_size_grid = int(case.num_prb)
        config.carrier.n_cell_id = int(case.n_cell_id)
        config.n_size_bwp = int(case.num_prb)
        config.num_antenna_ports = layers
        config.num_layers = layers
        config.precoding = "non-codebook"
        config.dmrs.config_type = int(case.dmrs_config_type)
        config.dmrs.length = int(case.dmrs_length)
        config.dmrs.additional_position = int(case.dmrs_additional_position)
        config.dmrs.num_cdm_groups_without_data = int(
            case.num_cdm_groups_without_data
        )
        config.dmrs.dmrs_port_set = ports
        config.tb.mcs_index = int(case.mcs_index)
        config.tb.mcs_table = int(case.mcs_table)
        config.n_rnti = int(100 + user_index)
        config.check_config()
        configs.append(config)
    return configs


def build_transmitter(case: NRCase, device: torch.device | str) -> tuple[Any, list[Any]]:
    from sionna.phy.nr import PUSCHTransmitter

    dev = normalize_device(device)
    configs = build_pusch_configs(case)
    transmitter = PUSCHTransmitter(
        configs,
        output_domain="freq",
        return_bits=True,
        device=str(dev),
    )
    return transmitter, configs


def _normalized_grid_coords(
    num_symbols: int,
    num_subcarriers: int,
    device: torch.device,
) -> torch.Tensor:
    time = torch.arange(num_symbols, dtype=torch.float32, device=device)
    freq = torch.arange(num_subcarriers, dtype=torch.float32, device=device)
    tt, ff = torch.meshgrid(time, freq, indexing="ij")
    coords = torch.stack([ff.reshape(-1), tt.reshape(-1)], dim=-1)
    if num_subcarriers > 1:
        coords[:, 0] = (coords[:, 0] - coords[:, 0].mean()) / float(
            num_subcarriers - 1
        )
    if num_symbols > 1:
        coords[:, 1] = (coords[:, 1] - coords[:, 1].mean()) / float(
            num_symbols - 1
        )
    return coords


def extract_nr_grid(
    transmitter: Any,
    pusch_configs: list[Any],
    device: torch.device | str,
) -> NRGridDescription:
    """Extract Sionna's exact PUSCH DMRS and data ordering.

    ``PUSCHPilotPattern.mask`` includes every RE reserved by the configured CDM
    groups. Its pilot vector can therefore contain zeros for REs reserved by a
    different port. We reconstruct the complete reserved grid first, then form
    the Bayesian observation matrix only from REs carrying nonzero DMRS energy.
    """
    dev = normalize_device(device)
    resource_grid = transmitter.resource_grid
    pattern = transmitter.pilot_pattern
    mask = pattern.mask.to(dev).bool()  # [U,L,S,F_eff]
    pilots = pattern.pilots.to(dev)
    if mask.ndim != 4:
        raise RuntimeError(f"Unexpected PUSCH pilot mask shape: {tuple(mask.shape)}")

    num_users = int(mask.shape[0])
    layers = int(mask.shape[1])
    if num_users != len(pusch_configs):
        raise RuntimeError("PUSCH pilot-mask transmitter count mismatch")
    if any(int(config.num_layers) != layers for config in pusch_configs):
        raise RuntimeError("PUSCH pilot-mask layer count mismatch")

    pilot_grid = torch.zeros(mask.shape, dtype=pilots.dtype, device=dev)
    for tx_index in range(num_users):
        for stream_index in range(layers):
            selector = mask[tx_index, stream_index].reshape(-1)
            values = pilots[tx_index, stream_index].reshape(-1)
            if int(selector.sum().item()) != int(values.numel()):
                raise RuntimeError("PUSCH pilot mask/pilot-vector length mismatch")
            flat = pilot_grid[tx_index, stream_index].reshape(-1)
            flat[selector] = values

    actual_pilot = pilot_grid.abs() > 0
    union_actual_pilot = actual_pilot.any(dim=(0, 1)).reshape(-1)
    pilot_idx = torch.nonzero(union_actual_pilot, as_tuple=False).flatten().long()
    if pilot_idx.numel() == 0:
        raise RuntimeError("No nonzero PUSCH DMRS symbols were found")

    stream_pilot_grid = pilot_grid.reshape(num_users * layers, *pilot_grid.shape[-2:])
    phi = stream_pilot_grid.reshape(num_users * layers, -1)[:, pilot_idx]
    if torch.any(torch.sum(torch.abs(phi) ** 2, dim=-1) <= 0):
        raise RuntimeError("At least one configured layer has no DMRS energy")

    effective = torch.as_tensor(
        np.asarray(resource_grid.effective_subcarrier_ind, dtype=np.int64),
        device=dev,
        dtype=torch.long,
    )
    type_grid_full = resource_grid.build_type_grid().to(dev)
    type_grid = torch.index_select(type_grid_full, -1, effective)
    data_masks = type_grid == 0
    reference_data = data_masks[0, 0]
    if not torch.all(data_masks == reference_data):
        raise RuntimeError(
            "Gate-1 bridge requires a common data-RE ordering across users and "
            "layers. Reserve every configured DMRS CDM group from data."
        )
    data_idx = torch.nonzero(reference_data.reshape(-1), as_tuple=False).flatten().long()
    reserved_mask = ~reference_data
    actual_mask = union_actual_pilot.reshape(reference_data.shape)
    reserved_zero_mask = reserved_mask & (~actual_mask)

    metadata: list[dict[str, Any]] = []
    for user_index, config in enumerate(pusch_configs):
        dmrs = config.dmrs
        cdm_groups = [int(x) for x in _to_list(dmrs.cdm_groups)]
        deltas = [int(x) for x in _to_list(dmrs.deltas)]
        wf = np.asarray(dmrs.w_f)
        wt = np.asarray(dmrs.w_t)
        ports = [int(x) for x in config.dmrs.dmrs_port_set]
        if not (
            len(ports) == layers
            and len(cdm_groups) == layers
            and len(deltas) == layers
            and wf.shape[-1] == layers
            and wt.shape[-1] == layers
        ):
            raise RuntimeError("PUSCH user/layer/port metadata length mismatch")
        for within_user_layer in range(layers):
            flat_layer = user_index * layers + within_user_layer
            metadata.append(
                {
                    "user_index": int(user_index),
                    "layer_index": int(flat_layer),
                    "within_user_layer_index": int(within_user_layer),
                    "dmrs_port": ports[within_user_layer],
                    "cdm_group": cdm_groups[within_user_layer],
                    "delta": deltas[within_user_layer],
                    "w_f": wf[:, within_user_layer].tolist(),
                    "w_t": wt[:, within_user_layer].tolist(),
                    "dmrs_config_type": int(dmrs.config_type),
                    "dmrs_length": int(dmrs.length),
                    "num_cdm_groups_without_data": int(
                        dmrs.num_cdm_groups_without_data
                    ),
                }
            )

    num_symbols = int(mask.shape[-2])
    num_effective = int(mask.shape[-1])
    coords = _normalized_grid_coords(num_symbols, num_effective, dev)
    bits_per_symbol = int(transmitter._num_bits_per_symbol)

    return NRGridDescription(
        coords=coords,
        pilot_idx=pilot_idx,
        data_idx=data_idx,
        phi=phi,
        pilot_grid=stream_pilot_grid,
        pilot_mask=actual_mask,
        reserved_mask=reserved_mask,
        reserved_zero_mask=reserved_zero_mask,
        data_mask=reference_data,
        effective_subcarrier_ind=effective,
        port_metadata=metadata,
        num_ofdm_symbols=num_symbols,
        num_effective_subcarriers=num_effective,
        fft_size=int(resource_grid.fft_size),
        bits_per_symbol=bits_per_symbol,
        num_users=num_users,
        num_layers_per_user=layers,
    )

def pilot_orthogonality_report(phi: torch.Tensor) -> dict[str, Any]:
    energy = torch.sum(torch.abs(phi) ** 2, dim=-1).real
    normalized = phi / torch.sqrt(energy.clamp_min(1e-12)).unsqueeze(-1)
    gram = normalized @ normalized.conj().transpose(-1, -2)
    identity = torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
    error = gram - identity
    offdiag = error - torch.diag_embed(torch.diagonal(error))
    return {
        "passed": bool(
            torch.max(torch.abs(offdiag)).item() < 2e-4
            and torch.max(torch.abs(torch.diagonal(error))).item() < 2e-4
        ),
        "max_abs_offdiagonal": float(torch.max(torch.abs(offdiag)).item()),
        "max_abs_diagonal_error": float(
            torch.max(torch.abs(torch.diagonal(error))).item()
        ),
        "energies": [float(x) for x in energy.detach().cpu().tolist()],
        "shape": list(phi.shape),
    }


def pilot_grid_match_report(
    transmitter: Any,
    grid: NRGridDescription,
    batch_size: int = 2,
) -> dict[str, Any]:
    """Verify exact Sionna DMRS, reserved-zero, and data-layer mapping."""
    x, b = transmitter(int(batch_size))
    coded = transmitter._tb_encoder(b)
    mapped = transmitter._mapper(coded)
    layered = transmitter._layer_mapper(mapped)

    effective = grid.effective_subcarrier_ind
    x_eff = torch.index_select(x, -1, effective)
    if x_eff.ndim != 5:
        raise RuntimeError(f"Unexpected PUSCH transmit-grid shape: {tuple(x_eff.shape)}")
    batch, users, layers, symbols, subcarriers = x_eff.shape
    if (
        users != grid.num_users
        or layers != grid.num_layers_per_user
        or symbols != grid.num_ofdm_symbols
        or subcarriers != grid.num_effective_subcarriers
    ):
        raise RuntimeError("PUSCH transmit grid and extracted grid disagree")
    x_stream = x_eff.reshape(batch, grid.num_streams, symbols, subcarriers)
    expected = grid.pilot_grid.unsqueeze(0).expand(batch, -1, -1, -1)

    actual_selector = (grid.pilot_grid.abs() > 0).unsqueeze(0).expand_as(x_stream)
    pilot_error = torch.abs(x_stream[actual_selector] - expected[actual_selector])

    reserved_selector = (
        grid.reserved_mask.view(1, 1, symbols, subcarriers).expand_as(x_stream)
    )
    expected_zero_selector = reserved_selector & (~actual_selector)
    reserved_zero_max = (
        float(torch.abs(x_stream[expected_zero_selector]).max().item())
        if expected_zero_selector.any()
        else 0.0
    )

    data_from_grid = x_stream.reshape(batch, grid.num_streams, -1)[..., grid.data_idx]
    expected_layered = layered.reshape(batch, grid.num_streams, grid.num_data_symbols)
    data_error = torch.abs(data_from_grid - expected_layered)

    expected_coded = (
        grid.num_data_symbols
        * grid.num_layers_per_user
        * grid.bits_per_symbol
    )
    return {
        "passed": bool(
            pilot_error.numel() > 0
            and float(pilot_error.max().item()) < 2e-6
            and reserved_zero_max < 2e-7
            and float(data_error.max().item()) < 2e-6
            and int(coded.shape[-1]) == int(expected_coded)
        ),
        "max_abs_pilot_grid_error": float(pilot_error.max().item()),
        "max_abs_reserved_zero_error": reserved_zero_max,
        "max_abs_data_layer_mapping_error": float(data_error.max().item()),
        "coded_bits_per_user": int(coded.shape[-1]),
        "expected_coded_bits_per_user": int(expected_coded),
        "reserved_zero_stream_re_count": int(expected_zero_selector[0].sum().item()),
        "globally_unused_reserved_re_count": int(grid.reserved_zero_mask.sum().item()),
        "x_shape": list(x.shape),
        "x_effective_shape": list(x_eff.shape),
        "layered_shape": list(layered.shape),
        "b_shape": list(b.shape),
    }


def codec_roundtrip(
    transmitter: Any,
    batch_size: int,
    device: torch.device | str,
    llr_magnitude: float = 20.0,
) -> dict[str, Any]:
    from sionna.phy.nr import TBDecoder

    dev = normalize_device(device)
    _, b = transmitter(int(batch_size))
    coded = transmitter._tb_encoder(b)
    decoder = TBDecoder(transmitter._tb_encoder, num_bp_iter=20, device=str(dev))
    llr = float(llr_magnitude) * (2.0 * coded - 1.0)
    b_hat, crc = decoder(llr)
    bit_errors = int(torch.sum(b_hat != b).item())
    return {
        "passed": bool(bit_errors == 0 and torch.all(crc).item()),
        "bit_errors": bit_errors,
        "crc_all_pass": bool(torch.all(crc).item()),
        "information_shape": list(b.shape),
        "coded_shape": list(coded.shape),
        "decoded_shape": list(b_hat.shape),
        "crc_shape": list(crc.shape),
    }


def _build_arrays(case: NRCase, device: torch.device) -> tuple[Any, Any]:
    from sionna.phy.channel.tr38901 import AntennaArray

    ut_array = AntennaArray(
        num_rows=1,
        num_cols=int(case.num_layers_per_user),
        polarization="single",
        polarization_type="V",
        antenna_pattern="omni",
        carrier_frequency=float(case.carrier_frequency_hz),
        device=str(device),
    )
    bs_array = AntennaArray(
        num_rows=1,
        num_cols=int(case.num_rx_ant),
        polarization="single",
        polarization_type="V",
        antenna_pattern="38.901",
        carrier_frequency=float(case.carrier_frequency_hz),
        device=str(device),
    )
    return ut_array, bs_array


def build_nr_context(case: NRCase, device: torch.device | str) -> NRContext:
    from sionna.phy.channel import OFDMChannel
    from sionna.phy.channel.tr38901 import CDL, UMa, UMi

    dev = normalize_device(device)
    transmitter, configs = build_transmitter(case, dev)
    grid = extract_nr_grid(transmitter, configs, dev)
    ut_array, bs_array = _build_arrays(case, dev)
    scenario = case.scenario.lower()

    if scenario == "umi":
        model = UMi(
            carrier_frequency=float(case.carrier_frequency_hz),
            o2i_model="low",
            ut_array=ut_array,
            bs_array=bs_array,
            direction="uplink",
            enable_pathloss=False,
            enable_shadow_fading=False,
            device=str(dev),
        )
    elif scenario == "uma":
        model = UMa(
            carrier_frequency=float(case.carrier_frequency_hz),
            o2i_model="low",
            ut_array=ut_array,
            bs_array=bs_array,
            direction="uplink",
            enable_pathloss=False,
            enable_shadow_fading=False,
            device=str(dev),
        )
    elif scenario.startswith("cdl-"):
        model = CDL(
            model=scenario.split("-", 1)[1].upper(),
            delay_spread=float(case.delay_spread_s),
            carrier_frequency=float(case.carrier_frequency_hz),
            ut_array=ut_array,
            bs_array=bs_array,
            direction="uplink",
            min_speed=float(case.speed_mps),
            max_speed=float(case.speed_mps),
            device=str(dev),
        )
    else:
        raise ValueError(f"No fading context for scenario {case.scenario}")

    channel = OFDMChannel(
        model,
        transmitter.resource_grid,
        add_awgn=True,
        normalize_channel=True,
        return_channel=True,
        device=str(dev),
    )
    return NRContext(
        case=case,
        transmitter=transmitter,
        pusch_configs=configs,
        grid=grid,
        channel_model=model,
        channel=channel,
        device=dev,
    )


def adapt_sionna_batch(
    transmitter: Any,
    grid: NRGridDescription,
    *,
    x: torch.Tensor,
    b: torch.Tensor,
    coded: torch.Tensor,
    y: torch.Tensor,
    h: torch.Tensor,
    no: torch.Tensor,
    ebno_db: float,
) -> NRBatch:
    """Flatten Sionna's [user,layer] axes into explicit receiver nodes."""
    effective = grid.effective_subcarrier_ind
    y_eff = torch.index_select(y[:, 0], -1, effective)  # [B,RX,S,F_eff]
    y_flat = y_eff.reshape(y_eff.shape[0], y_eff.shape[1], -1)

    # Sionna OFDM channel layout:
    # [B,num_rx=1,RX,num_tx,num_tx_ant/layer,S,F].
    if h.ndim != 7 or int(h.shape[1]) != 1:
        raise RuntimeError(f"Unexpected Sionna OFDM channel shape: {tuple(h.shape)}")
    if (
        int(h.shape[3]) != grid.num_users
        or int(h.shape[4]) != grid.num_layers_per_user
    ):
        raise RuntimeError(
            "Sionna channel user/layer dimensions disagree with the PUSCH grid: "
            f"h={tuple(h.shape)}, users={grid.num_users}, "
            f"layers={grid.num_layers_per_user}"
        )
    h_eff = torch.index_select(h[:, 0], -1, effective)
    # [B,RX,U,L,S,F] -> [B,U*L,RX,R]
    h_flat = h_eff.permute(0, 2, 3, 1, 4, 5).reshape(
        h.shape[0], grid.num_streams, h.shape[2], -1
    )

    x_eff = torch.index_select(x, -1, effective)
    if x_eff.ndim != 5:
        raise RuntimeError(f"Unexpected Sionna PUSCH grid shape: {tuple(x_eff.shape)}")
    x_stream = x_eff.reshape(
        x_eff.shape[0], grid.num_streams, grid.num_ofdm_symbols,
        grid.num_effective_subcarriers
    )

    expected_coded = (
        grid.num_data_symbols
        * grid.num_layers_per_user
        * grid.bits_per_symbol
    )
    if int(coded.shape[-1]) != expected_coded:
        raise RuntimeError(
            f"Coded-bit count {coded.shape[-1]} != D*L*Q={expected_coded}"
        )
    if int(coded.shape[1]) != grid.num_users:
        raise RuntimeError("Coded-bit transmitter dimension mismatch")
    # Mapper consumes Q consecutive bits and LayerMapper reshapes the symbol
    # sequence as [D,L] before transposing to [L,D]. Mirror that operation.
    coded_grid = (
        coded.reshape(
            coded.shape[0], grid.num_users, grid.num_data_symbols,
            grid.num_layers_per_user, grid.bits_per_symbol
        )
        .permute(0, 1, 3, 2, 4)
        .reshape(
            coded.shape[0], grid.num_streams, grid.num_data_symbols,
            grid.bits_per_symbol
        )
        .contiguous()
    )
    noise = torch.as_tensor(no, dtype=torch.float32, device=y.device).mean()
    return NRBatch(
        y=y_flat,
        h=h_flat,
        information_bits=b,
        coded_bits=coded_grid,
        x_grid=x_stream,
        noise_var=noise,
        ebno_db=float(ebno_db),
        phi=grid.phi,
        pilot_idx=grid.pilot_idx,
        data_idx=grid.data_idx,
        port_metadata=grid.port_metadata,
        raw_y=y,
        raw_h=h,
    )


def fixed_cardinality_mask(scores: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Select per-row top scores using exactly the reference number of edges."""
    if scores.shape != reference.shape:
        raise ValueError("scores and reference graph must have identical shapes")
    n = int(scores.shape[-1])
    eye = torch.eye(n, dtype=torch.bool, device=scores.device).view(1, 1, n, n)
    clean = scores.detach().real.masked_fill(eye, -torch.inf)
    counts = reference.sum(dim=-1).long()
    order = torch.argsort(clean, dim=-1, descending=True)
    ranks = torch.arange(n, device=scores.device).view(1, 1, 1, n)
    keep_sorted = ranks < counts.unsqueeze(-1)
    mask = torch.zeros_like(reference, dtype=torch.bool)
    mask.scatter_(-1, order, keep_sorted)
    return mask & (~eye)


def mask_as_kappa(mask: torch.Tensor) -> torch.Tensor:
    # With edge_mass=1, coupling_selection_mask retains every positive score.
    return mask.to(torch.float32)


def diagonal_covariance(local_cov: torch.Tensor) -> torch.Tensor:
    n = int(local_cov.shape[0])
    result = torch.zeros_like(local_cov)
    index = torch.arange(n, device=local_cov.device)
    result[index, index] = local_cov[index, index]
    return result


def homoscedastic_covariance(local_cov: torch.Tensor) -> torch.Tensor:
    n = int(local_cov.shape[0])
    diagonal = torch.stack([local_cov[i, i].real for i in range(n)], dim=0)
    scalar = diagonal.mean().clamp_min(1e-8)
    result = torch.zeros_like(local_cov)
    index = torch.arange(n, device=local_cov.device)
    result[index, index] = scalar.to(result.dtype)
    return result


class NRBayesRouteBridge(nn.Module):
    """BayesRoute posterior and detector operating on a real NR PUSCH grid."""

    def __init__(
        self,
        grid: NRGridDescription,
        *,
        num_streams: int,
        rank: int = 16,
        bank_rank: int = 24,
        detector_iterations: int = 4,
        edge_mass: float = 0.8,
        length_f: float = 2.0,
        length_t: float = 1.0,
        operator_seed: int = 28001,
    ) -> None:
        super().__init__()
        self.grid = grid
        self.num_streams = int(num_streams)
        if self.num_streams != int(grid.num_streams):
            raise ValueError(
                f"num_streams={self.num_streams} does not match grid={grid.num_streams}"
            )
        self.edge_mass = float(edge_mass)
        self.posterior = LowRankPosteriorOperator(
            coords=grid.coords,
            pilot_idx=grid.pilot_idx,
            n_layers=self.num_streams,
            rank=int(rank),
            length_f=float(length_f),
            length_t=float(length_t),
            seed=int(operator_seed),
            bank_rank=int(bank_rank),
        )
        self.detector = BayesRouteDetector(
            bits_per_symbol=int(grid.bits_per_symbol),
            n_iter=int(detector_iterations),
            use_uncertainty=True,
        )

    def forward_variant(
        self,
        batch: NRBatch,
        variant: str = "proposed",
        *,
        random_seed: int = 0,
        cache: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if cache is None:
            cache = {}
        posterior = cache.get("posterior")
        if posterior is None:
            posterior = self.posterior(
                batch.y[..., batch.pilot_idx],
                batch.phi,
                batch.noise_var,
            )
            cache["posterior"] = posterior
        full_cov = posterior.local_cov
        zero_cov = torch.zeros_like(full_cov)
        diag_cov = cache.get("diagonal_covariance")
        if diag_cov is None:
            diag_cov = diagonal_covariance(full_cov)
            cache["diagonal_covariance"] = diag_cov

        full_kappa = cache.get("full_kappa")
        if full_kappa is None:
            full_kappa = coupling_matrix(
                posterior.mean.detach(),
                full_cov.detach(),
                batch.data_idx,
                batch.noise_var.detach(),
            )
            cache["full_kappa"] = full_kappa
        reference = cache.get("reference_graph")
        if reference is None:
            reference = coupling_selection_mask(full_kappa, self.edge_mass)
            cache["reference_graph"] = reference

        detector_cov = full_cov
        use_uncertainty = True
        selected = reference

        if variant == "proposed":
            pass
        elif variant == "uncertainty_off_fixed_graph":
            use_uncertainty = False
        elif variant == "diagonal_posterior_fixed_graph":
            detector_cov = diag_cov
        elif variant == "mean_only_graph_fixed_cardinality":
            mean_kappa = coupling_matrix(
                posterior.mean.detach(),
                zero_cov,
                batch.data_idx,
                batch.noise_var.detach(),
            )
            selected = fixed_cardinality_mask(mean_kappa, reference)
        elif variant == "random_graph_fixed_cardinality":
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int(random_seed))
            random_scores = torch.rand(
                full_kappa.shape, generator=generator, dtype=torch.float32
            ).to(full_kappa.device)
            selected = fixed_cardinality_mask(random_scores, reference)
        elif variant == "full_graph":
            n = int(full_kappa.shape[-1])
            eye = torch.eye(n, dtype=torch.bool, device=full_kappa.device).view(
                1, 1, n, n
            )
            selected = torch.ones_like(reference) & (~eye)
        elif variant == "graph_off":
            selected = torch.zeros_like(reference)
        else:
            raise ValueError(f"Unknown Gate-1 bridge variant: {variant}")

        kappa_for_detector = mask_as_kappa(selected)
        bit_logits, symbol_logits, x_mean, x_var, graph_mask = self.detector(
            batch.y,
            posterior.mean,
            detector_cov,
            batch.data_idx,
            batch.noise_var,
            kappa=kappa_for_detector,
            edge_mass=1.0,
            use_uncertainty=use_uncertainty,
        )
        if not torch.equal(graph_mask, selected):
            raise RuntimeError("Fixed-cardinality graph was not preserved by detector")
        return {
            "bit_logits": bit_logits,
            "symbol_logits": symbol_logits,
            "x_mean": x_mean,
            "x_var": x_var,
            "posterior": posterior,
            "graph_mask": graph_mask,
            "reference_graph_mask": reference,
            "edge_density": edge_density(graph_mask),
            "reference_edge_density": edge_density(reference),
            "variant": variant,
        }

    def forward_variants(
        self,
        batch: NRBatch,
        variants: Iterable[str],
        *,
        random_seed: int = 0,
    ) -> dict[str, dict[str, Any]]:
        cache: dict[str, Any] = {}
        # Use the same explicit random seed regardless of which subset remains
        # after resumption. Only the random-graph control consumes this seed.
        return {
            name: self.forward_variant(
                batch, name, random_seed=int(random_seed), cache=cache
            )
            for name in variants
        }

    def forward(self, batch: NRBatch) -> dict[str, Any]:
        return self.forward_variant(batch, "proposed")


def transfer_operator_parameters(source: nn.Module, target: nn.Module) -> dict[str, Any]:
    """Transfer only learned scalar/spectral parameters across NR configurations."""
    source_state = source.state_dict()
    target_state = target.state_dict()
    copied: list[str] = []
    for name in ("posterior.raw_weights", "posterior.log_noise_scale"):
        if name in source_state and name in target_state:
            if source_state[name].shape != target_state[name].shape:
                raise RuntimeError(
                    f"Cannot transfer {name}: {source_state[name].shape} != "
                    f"{target_state[name].shape}"
                )
            target_state[name] = source_state[name].detach().clone()
            copied.append(name)
    target.load_state_dict(target_state, strict=True)
    return {"copied": copied, "passed": len(copied) == 2}


def coded_bit_metrics(logits: torch.Tensor, bits: torch.Tensor) -> dict[str, float]:
    with torch.no_grad():
        bits = bits.float()
        hard = (logits >= 0).float()
        probs = torch.sigmoid(logits)
        return {
            "coded_ber": float((hard != bits).float().mean().item()),
            "coded_bit_nll": float(
                F.binary_cross_entropy_with_logits(logits, bits).item()
            ),
            "coded_brier": float(torch.mean((probs - bits) ** 2).item()),
        }


def channel_metrics(output: dict[str, Any], batch: NRBatch) -> dict[str, float]:
    posterior = output["posterior"]
    mean = posterior.mean[..., batch.data_idx]
    truth = batch.h[..., batch.data_idx]
    var = posterior.var_diag[None, :, None, batch.data_idx].to(mean.device)
    error = torch.abs(mean - truth) ** 2
    nmse = error.mean() / torch.abs(truth).square().mean().clamp_min(1e-8)
    nll = (error / var.clamp_min(1e-8) + torch.log(var.clamp_min(1e-8))).mean()
    threshold = -math.log(0.05) * var
    coverage = (error <= threshold).float().mean()
    return {
        "channel_nmse": float(nmse.real.item()),
        "channel_marginal_nll": float(nll.real.item()),
        "channel_coverage95": float(coverage.item()),
    }


def decode_bridge(
    transmitter: Any,
    output: dict[str, Any],
    information_bits: torch.Tensor,
    *,
    num_bp_iter: int = 20,
    device: torch.device | str,
    decoder: Any | None = None,
    layer_demapper: Any | None = None,
) -> dict[str, Any]:
    """Layer-demap BayesRoute coded-bit LLRs and run Sionna TB decoding."""
    from sionna.phy.nr import LayerDemapper, TBDecoder

    dev = normalize_device(device)
    logits = output["bit_logits"]
    batch, streams, data_symbols, bits_per_symbol = logits.shape
    num_users = int(information_bits.shape[1])
    num_layers = int(transmitter._num_layers)
    if streams != num_users * num_layers:
        raise RuntimeError(
            f"Bridge stream count {streams} != users*layers={num_users*num_layers}"
        )
    # LayerDemapper expects [B,U,L,D*Q] for this PUSCH transmitter.
    layer_llr = (
        logits.reshape(
            batch, num_users, num_layers, data_symbols, bits_per_symbol
        )
        .reshape(batch, num_users, num_layers, data_symbols * bits_per_symbol)
        .contiguous()
    )
    if layer_demapper is None:
        layer_demapper = LayerDemapper(
            transmitter._layer_mapper,
            num_bits_per_symbol=int(bits_per_symbol),
            device=str(dev),
        )
    codeword_llr = layer_demapper(layer_llr)
    if isinstance(codeword_llr, list):
        raise RuntimeError("Gate-1 PUSCH expects a single codeword per transmitter")
    if decoder is None:
        decoder = TBDecoder(
            transmitter._tb_encoder,
            num_bp_iter=int(num_bp_iter),
            device=str(dev),
        )
    b_hat, crc = decoder(codeword_llr)
    bit_error = (b_hat != information_bits).float()
    block_error = bit_error.reshape(bit_error.shape[0], bit_error.shape[1], -1).any(-1)
    return {
        "information_ber": float(bit_error.mean().item()),
        "tbler": float(block_error.float().mean().item()),
        "crc_failure_rate": float((~crc.bool()).float().mean().item()),
        "layer_llr_shape": list(layer_llr.shape),
        "codeword_llr_shape": list(codeword_llr.shape),
        "decoded_shape": list(b_hat.shape),
        "crc_shape": list(crc.shape),
        "b_hat": b_hat,
        "crc": crc,
    }


def standard_receiver(
    context: NRContext,
    *,
    perfect_csi: bool = False,
    kbest_k: int | None = None,
    return_crc: bool = True,
) -> Any:
    from sionna.phy.mimo import StreamManagement
    from sionna.phy.nr import PUSCHReceiver
    from sionna.phy.ofdm import KBestDetector

    detector = None
    stream_management = None
    kbest_compatibility = None
    if kbest_k is not None:
        # BayesRoute Gate-1 compatibility patch:
        # Sionna 2.0.1 compiles only an equal+any helper in List2LLRSimple.
        # Alliance's CUDA PyTorch build has no Triton, so use the exact eager
        # expression. K-best candidates, distances, LLR equations, and outputs
        # are unchanged.
        from .sionna_kbest_compat import configure_sionna_kbest_compat

        kbest_compatibility = configure_sionna_kbest_compat(force_eager=True)
        if not kbest_compatibility.get("passed", False):
            raise RuntimeError(
                f"Sionna K-best compatibility self-test failed: {kbest_compatibility}"
            )

        rx_tx_association = np.ones([1, context.case.num_users], dtype=bool)
        stream_management = StreamManagement(
            rx_tx_association, int(context.case.num_layers_per_user)
        )
        detector = KBestDetector(
            output="bit",
            num_streams=int(context.case.num_streams),
            k=int(kbest_k),
            resource_grid=context.transmitter.resource_grid,
            stream_management=stream_management,
            constellation_type="qam",
            num_bits_per_symbol=context.transmitter._num_bits_per_symbol,
            device=str(context.device),
        )
    receiver = PUSCHReceiver(
        context.transmitter,
        channel_estimator="perfect" if perfect_csi else None,
        mimo_detector=detector,
        return_tb_crc_status=bool(return_crc),
        stream_management=stream_management,
        input_domain="freq",
        device=str(context.device),
    )
    if kbest_compatibility is not None:
        receiver._bayesroute_kbest_compatibility = kbest_compatibility
    return receiver


def run_standard_receiver(
    receiver: Any,
    batch: NRBatch,
    information_bits: torch.Tensor,
    *,
    perfect_csi: bool,
) -> dict[str, Any]:
    if perfect_csi:
        result = receiver(batch.raw_y, batch.noise_var, batch.raw_h)
    else:
        result = receiver(batch.raw_y, batch.noise_var)
    if isinstance(result, tuple):
        b_hat, crc = result
    else:
        b_hat = result
        crc = torch.ones(
            b_hat.shape[:-1] + (1,), dtype=torch.bool, device=b_hat.device
        )
    bit_error = (b_hat != information_bits).float()
    block_error = bit_error.reshape(bit_error.shape[0], bit_error.shape[1], -1).any(-1)
    return {
        "information_ber": float(bit_error.mean().item()),
        "tbler": float(block_error.float().mean().item()),
        "crc_failure_rate": float((~crc.bool()).float().mean().item()),
        "decoded_shape": list(b_hat.shape),
        "crc_shape": list(crc.shape),
    }


def graph_cardinality_report(outputs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    reference = outputs["proposed"]["graph_mask"]
    required = [
        "uncertainty_off_fixed_graph",
        "diagonal_posterior_fixed_graph",
        "mean_only_graph_fixed_cardinality",
        "random_graph_fixed_cardinality",
    ]
    counts = reference.sum(dim=-1)
    same: dict[str, bool] = {}
    for name in required:
        same[name] = bool(
            torch.equal(outputs[name]["graph_mask"].sum(dim=-1), counts)
        )
    same["uncertainty_off_exact_mask"] = bool(
        torch.equal(outputs["uncertainty_off_fixed_graph"]["graph_mask"], reference)
    )
    same["diagonal_exact_mask"] = bool(
        torch.equal(outputs["diagonal_posterior_fixed_graph"]["graph_mask"], reference)
    )
    return {
        "passed": all(same.values()),
        "checks": same,
        "reference_edge_density": float(edge_density(reference).item()),
        "reference_count_min": int(counts.min().item()),
        "reference_count_max": int(counts.max().item()),
    }


def posterior_psd_report(output: dict[str, Any]) -> dict[str, Any]:
    covariance = output["posterior"].latent_cov
    hermitian = torch.max(torch.abs(covariance - covariance.conj().transpose(-1, -2)))
    eig = torch.linalg.eigvalsh(covariance.to(torch.complex128)).real
    finite = bool(
        torch.isfinite(output["posterior"].mean).all().item()
        and torch.isfinite(covariance).all().item()
        and torch.isfinite(output["bit_logits"]).all().item()
    )
    return {
        "passed": bool(finite and eig.min().item() >= -1e-7 and hermitian.item() < 1e-5),
        "finite": finite,
        "min_eigenvalue": float(eig.min().item()),
        "max_hermitian_error": float(hermitian.item()),
        "posterior_mean_shape": list(output["posterior"].mean.shape),
        "bit_logits_shape": list(output["bit_logits"].shape),
    }


def package_contract_signature(payload: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
