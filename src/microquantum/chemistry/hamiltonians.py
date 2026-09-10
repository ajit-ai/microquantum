"""Molecular Hamiltonians for quantum chemistry calculations.

Provides pre-computed molecular Hamiltonians for common molecules
(H2, LiH) as Pauli operator sums, along with a base class for
custom molecular Hamiltonians.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..core.pauli import PauliString, PauliSum
from ..core.state import StateVector


@dataclass
class MolecularHamiltonian:
    """Base class for molecular Hamiltonians.

    A molecular Hamiltonian is represented as a Pauli sum:
        H = sum_i c_i * P_i

    where c_i are real coefficients and P_i are Pauli strings.

    Attributes:
        name: Name of the molecule.
        num_qubits: Number of qubits needed.
        num_electrons: Number of electrons.
        hamiltonian: PauliSum representation.
        num_spatial_orbitals: Number of spatial orbitals.
        bond_length: Equilibrium bond length in Angstroms.
        ground_state_energy: Exact ground state energy (if known).
        nuclear_repulsion: Nuclear repulsion energy.
    """
    name: str
    num_qubits: int
    num_electrons: int
    hamiltonian: PauliSum
    num_spatial_orbitals: int = 0
    bond_length: float = 0.0
    ground_state_energy: Optional[float] = None
    nuclear_repulsion: float = 0.0

    def __post_init__(self) -> None:
        if self.num_spatial_orbitals == 0:
            self.num_spatial_orbitals = self.num_qubits

    @property
    def num_terms(self) -> int:
        """Number of Pauli terms in the Hamiltonian."""
        return len(self.hamiltonian.terms)

    def energy(self, statevector: StateVector) -> float:
        """Compute expectation value of H for a given state.

        Args:
            statevector: Quantum state vector.

        Returns:
            Expectation value <psi|H|psi>.
        """
        return self.hamiltonian.expectation(statevector)

    def __repr__(self) -> str:
        return (
            f"MolecularHamiltonian(name={self.name!r}, "
            f"qubits={self.num_qubits}, terms={self.num_terms})"
        )


class H2Hamiltonian(MolecularHamiltonian):
    """Hydrogen molecule (H2) Hamiltonian in STO-3G basis.

    2-qubit Hamiltonian from Jordan-Wigner transformation of the
    second-quantized H2 Hamiltonian at equilibrium bond length.

    Reference: Peruzzo et al., "A variational eigenvalue solver on
    a photonic quantum processor", Nat. Commun. 5, 4213 (2014).
    """

    def __init__(self, bond_length: float = 0.735) -> None:
        self._bond_length = bond_length

        # Pauli decomposition of H2 at equilibrium
        # H = g0*I + g1*Z0 + g2*Z1 + g3*Z0Z1 + g4*X0X1 + g5*Y0Y1
        g0 = -0.4804
        g1 = 0.3435
        g2 = -0.4347
        g3 = 0.5716
        g4 = 0.0910
        g5 = 0.0910

        terms = [
            PauliString("II", g0),
            PauliString("ZI", g1),
            PauliString("IZ", g2),
            PauliString("ZZ", g3),
            PauliString("XX", g4),
            PauliString("YY", g5),
        ]
        hamiltonian = PauliSum(terms)
        nuclear_repulsion = 0.7199689944499149

        super().__init__(
            name="H2",
            num_qubits=2,
            num_electrons=2,
            hamiltonian=hamiltonian,
            num_spatial_orbitals=2,
            bond_length=bond_length,
            ground_state_energy=-1.857275030222,
            nuclear_repulsion=nuclear_repulsion,
        )


class LiHHamiltonian(MolecularHamiltonian):
    """Lithium hydride (LiH) Hamiltonian in STO-3G basis.

    6-qubit Hamiltonian from Jordan-Wigner transformation.
    This is a larger molecule that demonstrates the scaling
    of VQE algorithms.
    """

    def __init__(self, bond_length: float = 1.6) -> None:
        self._bond_length = bond_length

        # Simplified LiH Hamiltonian terms (reduced from full form)
        g0 = -7.8618
        terms = [
            PauliString("IIIIII", g0),
            PauliString("ZIIIII", 0.1659),
            PauliString("IZIIII", 0.1202),
            PauliString("IIZIII", -0.0454),
            PauliString("IIIZII", -0.0454),
            PauliString("IIIIZI", 0.1202),
            PauliString("IIIIIZ", 0.1659),
            PauliString("ZIZIII", 0.0228),
            PauliString("IIZIZI", 0.0228),
            PauliString("XXIIII", 0.0312),
            PauliString("YYIIII", 0.0312),
        ]
        hamiltonian = PauliSum(terms)

        super().__init__(
            name="LiH",
            num_qubits=6,
            num_electrons=4,
            hamiltonian=hamiltonian,
            num_spatial_orbitals=6,
            bond_length=bond_length,
            ground_state_energy=-7.8822,
            nuclear_repulsion=3.2260,
        )
