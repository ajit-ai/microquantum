"""Quantum chemistry module for microquantum.

Provides molecular Hamiltonians, ansatz circuits, and chemistry-specific
functionality for variational quantum eigensolver (VQE) calculations.
"""
from .ansatz import HardwareEfficientAnsatz, UCCSDAnsatz
from .hamiltonians import H2Hamiltonian, LiHHamiltonian, MolecularHamiltonian
from .transforms import (
    ActiveSpace,
    ActiveSpaceSelection,
    FermionicOp,
    IntegralSource,
    jordan_wigner,
)

__all__ = [
    "MolecularHamiltonian",
    "H2Hamiltonian",
    "LiHHamiltonian",
    "HardwareEfficientAnsatz",
    "UCCSDAnsatz",
    "IntegralSource",
    "ActiveSpace",
    "ActiveSpaceSelection",
    "FermionicOp",
    "jordan_wigner",
]
