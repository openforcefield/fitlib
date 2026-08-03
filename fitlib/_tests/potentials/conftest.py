import copy

import pytest
import torch

import fitlib._tests.utils


@pytest.fixture(scope="module")
def _etoh_water_system() -> tuple[
    fitlib.TensorSystem, fitlib.TensorForceField, torch.Tensor, torch.Tensor
]:
    openmm_unit = pytest.importorskip("openmm.unit")
    fitlib_mm = pytest.importorskip("fitlib.mm")

    system, force_field = fitlib._tests.utils.system_from_smiles(
        ["CCO", "O"], [67, 123]
    )
    coords, box_vectors = fitlib_mm.generate_system_coords(system, None)

    return (
        system,
        force_field,
        torch.tensor(coords.value_in_unit(openmm_unit.angstrom), dtype=torch.float32),
        torch.tensor(
            box_vectors.value_in_unit(openmm_unit.angstrom), dtype=torch.float32
        ),
    )


@pytest.fixture()
def etoh_water_system(
    _etoh_water_system,
) -> tuple[fitlib.TensorSystem, fitlib.TensorForceField, torch.Tensor, torch.Tensor]:
    """Creates a system of ethanol and water."""

    return copy.deepcopy(_etoh_water_system)
