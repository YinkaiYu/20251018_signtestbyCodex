import numpy as np
import pytest

from worldline_qmc import worldline


def test_permutation_parity_identity() -> None:
    perm = worldline.PermutationState.identity(4)
    assert perm.parity() == 1


def test_permutation_parity_transposition() -> None:
    perm = worldline.PermutationState(np.array([1, 0, 2], dtype=np.int64))
    assert perm.parity() == -1


def test_worldline_enforces_pauli_on_init() -> None:
    trajectories = np.array([[0, 1], [2, 2]], dtype=np.int64)
    with pytest.raises(ValueError):
        worldline.Worldline(trajectories)


def test_worldline_update_rejects_duplicate() -> None:
    traj = np.array([[0, 1], [2, 3]], dtype=np.int64)
    wl = worldline.Worldline(traj)
    with pytest.raises(ValueError):
        wl.update_momentum(time_slice=0, particle=0, new_k=1)


def test_worldline_update_and_occupancy() -> None:
    traj = np.array([[0, 1], [2, 3]], dtype=np.int64)
    wl = worldline.Worldline(traj)
    wl.update_momentum(time_slice=0, particle=0, new_k=2)
    assert np.array_equal(wl.occupancy(0), np.array([2, 1]))
    assert wl.trajectories[0, 0] == 2


def test_unflatten_round_trip() -> None:
    index = worldline.flatten_momentum_index(1, 2, lattice_size=4)
    ix, iy = worldline.unflatten_momentum_index(index, lattice_size=4)
    assert (ix, iy) == (1, 2)
