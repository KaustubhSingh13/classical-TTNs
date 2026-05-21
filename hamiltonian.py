import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh

def paulis():
    sx = np.array([[0, 1], [1, 0]], dtype=float)
    sz = np.array([[1, 0], [0, -1]], dtype=float)
    I  = np.eye(2)
    return sx, sz, I

class Hamiltonian:
    def __init__(self, n_sites):
        self.n_sites = n_sites
        self._single_site_terms = []
        self._two_site_terms    = []
        self._matrix_cache      = None

    def _invalidate_cache(self):
        self._matrix_cache = None

    def add_single_site(self, site, op):
        self._single_site_terms.append((site, op))
        self._invalidate_cache()

    def add_two_site(self, i, j, op_i, op_j):
        self._two_site_terms.append((op_i, op_j, i, j)) 
        self._invalidate_cache()

    def single_site_terms(self):
        return self._single_site_terms

    def two_site_terms(self):
        return self._two_site_terms

    def clear(self):
        self._single_site_terms.clear()
        self._two_site_terms.clear()
        self._invalidate_cache()

    def _embed_single(self, site, op):
        """Build sparse single-site operator via sparse Kronecker products."""
        I = sparse.eye(2, format="csr")
        op_sp = sparse.csr_matrix(op)
        out = sparse.eye(1, format="csr")
        for k in range(self.n_sites):
            out = sparse.kron(out, op_sp if k == site else I, format="csr")
        return out 

    def _embed_two(self, i, j, op_i, op_j):
        """Build sparse two-site operator via sparse Kronecker products."""
        I    = sparse.eye(2, format="csr")
        op_i_sp = sparse.csr_matrix(op_i)
        op_j_sp = sparse.csr_matrix(op_j)
        out = sparse.eye(1, format="csr")
        for k in range(self.n_sites):
            if k == i:
                out = sparse.kron(out, op_i_sp, format="csr")
            elif k == j:
                out = sparse.kron(out, op_j_sp, format="csr")
            else:
                out = sparse.kron(out, I, format="csr")
        return out

    def to_matrix(self, dense=False):
        """Return the Hamiltonian as a sparse CSR matrix (or dense array)."""
        if self._matrix_cache is None:
            dim = 2 ** self.n_sites
            H = sparse.csr_matrix((dim, dim), dtype=float)
            for site, op in self._single_site_terms:
                H = H + self._embed_single(site, op)
            for op_i, op_j, i, j in self._two_site_terms:
                H = H + self._embed_two(i, j, op_i, op_j)
            self._matrix_cache = H
        return self._matrix_cache.toarray() if dense else self._matrix_cache
    def exact_diagonalization(self, return_eigenvectors=False, only_ground=True):
        H = self.to_matrix()   # sparse
        dim = H.shape[0]

        # Below this size, dense eigh is exact and fast enough
        DENSE_THRESHOLD = 2**12   # ~12 sites

        if dim <= DENSE_THRESHOLD:
            # Dense path: always exact, no convergence concerns
            evals, evecs = np.linalg.eigh(H.toarray())
            if only_ground:
                return (evals[0], evecs[:, 0]) if return_eigenvectors else evals[0]
            else:
                return (evals, evecs) if return_eigenvectors else evals

        # Sparse path for large systems
        if only_ground:
            # k=2 gives ARPACK a larger subspace to work in, significantly
            # improving convergence near degeneracies / small gaps.
            # ncv >> k adds more Lanczos vectors.
            # v0 fixed seed makes results reproducible.
            ncv = min(max(50, 4 * 2), dim - 1)
            v0  = np.ones(dim) / np.sqrt(dim)
            evals, evecs = eigsh(
                H, k=2, which="SA",
                tol=1e-12, ncv=ncv, maxiter=10 * dim, v0=v0
            )
            # evals sorted ascending; take index 0
            idx = np.argmin(evals)
            if return_eigenvectors:
                return evals[idx], evecs[:, idx]
            return evals[idx]

        else:
            # Full spectrum on sparse matrix: use LOBPCG or warn user
            # eigh on a 65536x65536 matrix needs ~32 GB — not feasible at n=16
            if dim > DENSE_THRESHOLD:
                raise ValueError(
                    f"Full diagonalization of dim={dim} requires ~"
                    f"{dim**2 * 8 / 1e9:.1f} GB. "
                    "Use only_ground=True for large systems, or reduce n_sites. Also not stable."
                )
            evals, evecs = np.linalg.eigh(H.toarray())
        return (evals, evecs) if return_eigenvectors else evals
    '''
    def exact_diagonalization(self, return_eigenvectors=False, only_ground=True):
        H = self.to_matrix()

        if only_ground:
            
            evals, evecs = eigsh(H, k=1, which="SA")
            if return_eigenvectors:
                return evals[0], evecs[:, 0]   
            return evals[0]
        else:
           
            H_dense = self.to_matrix(dense=True)
            evals, evecs = np.linalg.eigh(H_dense)
            if return_eigenvectors:
                return evals, evecs
            return evals
    '''

'''

class TFIsing(Hamiltonian):
    def __init__(self, n_sites, n_legs, J=1.0, h=1.0, periodic=False):
        super().__init__(n_sites)
        
        assert n_sites <= n_legs, 'Need n_sites to be <= number of legs'

        n_left_empty = n_legs - n_sites
        sx, sz, I = paulis()
        # transverse field
        for i in range(n_sites):
            self.add_single_site(i, -h * sx)
        for i in range(n_sites, n_legs):
            self.add_single_site(i, I)
        # ZZ interactions
        for i in range(n_sites - 1):
            self.add_two_site(i, i+1, -J * sz, sz)
'''

class TFIsing(Hamiltonian):
    """
    Transverse-field Ising model:
        H = -J Σ_{i=0}^{n_sites-2} Z_i Z_{i+1}
            [-J Z_{n_sites-1} Z_0   if periodic]
            -h Σ_{i=0}^{n_sites-1} X_i

    Parameters
    ----------
    n_sites  : number of physical spins.
    n_legs   : total legs on the TTN node (>= n_sites).
               Empty legs [n_sites, n_legs) are left untouched —
               apply_H_eff treats absent sites as identity by isometry.
    J, h     : ZZ coupling and transverse field.
    periodic : add the (n_sites-1, 0) boundary bond.
    """
    def __init__(self, n_sites: int, n_legs: int,
                 J: float = 1.0,h =1, periodic: bool = False):
        super().__init__(n_legs)          # n_legs, not n_sites
        assert n_sites >= 1,       "Need at least one physical site."
        assert n_legs  >= n_sites, "n_legs must be >= n_sites."

        sx, sz, _ = paulis()

        for i in range(n_sites):
            self.add_single_site(i, -h * sx)

        for i in range(n_sites - 1):
            self.add_two_site(i, i + 1, -J * sz, sz)

        if periodic and n_sites > 2:
            self.add_two_site(n_sites - 1, 0, -J * sz, sz)

            
class heisenberg(Hamiltonian):
    """
    Spin-1/2 Heisenberg model (nearest-neighbour, open or periodic chain):

        H = J Σ_{<i,j>} (X_i X_j + Y_i Y_j + Z_i Z_j)
            - h Σ_i Z_i

    The XY part is stored using the exact real identity
        X_i X_j + Y_i Y_j  =  2 (S⁺_i S⁻_j + S⁻_i S⁺_j)
    so that ALL operators are real-valued and H_eff remains a real symmetric
    matrix — no complex casting errors.

    Parameters
    ----------
    n_sites  : number of physical spins.
    n_legs   : total TTN legs (>= n_sites).
    J        : exchange coupling.
               J > 0  →  antiferromagnet (AFM).
               J < 0  →  ferromagnet (FM).
    h        : longitudinal field (−h Σ Z_i term).  Default 0.
    periodic : include the (n_sites-1, 0) boundary bond.
    """

    def __init__(self, n_sites: int, n_legs: int,
                 J: float = 1.0, h: float = 0.0,
                 periodic: bool = False):
        super().__init__(n_legs, n_phys_sites=n_sites)
        assert n_sites >= 1,       "Need at least one physical site."
        assert n_legs  >= n_sites, "n_legs must be >= n_sites."

        sx, sz, _ = paulis()
        sp, sm    = raising_lowering()

        # Longitudinal field (zero by default for pure Heisenberg)
        if h != 0.0:
            for i in range(n_sites):
                self.add_single_site(i, -h * sz)

        # Nearest-neighbour bonds: X⊗X + Y⊗Y + Z⊗Z
        # Written as  2(S⁺⊗S⁻ + S⁻⊗S⁺) + Z⊗Z  — all real operators.
        bonds = list(range(n_sites - 1))
        if periodic and n_sites > 2:
            bonds.append(n_sites - 1)   # wraps: pair (n_sites-1, 0)

        for i in bonds:
            j = (i + 1) % n_sites       # handles periodic wrap correctly
            self.add_two_site(i, j,  J * sz,      sz)      # Z⊗Z
            self.add_two_site(i, j,  2 * J * sp,  sm)      # 2J S⁺⊗S⁻
            self.add_two_site(i, j,  2 * J * sm,  sp)      # 2J S⁻⊗S⁺


class Free(Hamiltonian):
    """
    Non-interacting spins in a longitudinal field:
        H = -h Σ_i Z_i
    """

    def __init__(self, n_sites: int, n_legs: int = None, h: float = 1.0):
        n_legs = n_legs if n_legs is not None else n_sites
        super().__init__(n_legs, n_phys_sites=n_sites)
        assert n_legs >= n_sites, "n_legs must be >= n_sites."

        _, sz, _ = paulis()
        for i in range(n_sites):
            self.add_single_site(i, -h * sz)

class lmg(Hamiltonian):
    """
    Lipkin-Meshkov-Glick model:
        H = -(J / N) Σ_{i < j} (X_i X_j + γ Y_i Y_j)  -  h Σ_i Z_i

    The 1/N prefactor makes the model extensive (energy ∝ N in both phases).

    The XY anisotropy term is stored with REAL operators only using:
        X_i X_j + γ Y_i Y_j
          = (1−γ)(S⁺_i S⁺_j + S⁻_i S⁻_j)  +  (1+γ)(S⁺_i S⁻_j + S⁻_i S⁺_j)
    so H_eff is always real symmetric — no complex casting errors.

    Parameters
    ----------
    n_sites : number of physical spins (N in the 1/N prefactor).
    n_legs  : total TTN legs (>= n_sites).
    J       : all-to-all coupling strength.
    h       : longitudinal field (Σ Z_i term).
    gamma   : XY anisotropy.
                γ = 0  →  pure XX model.
                γ = 1  →  isotropic XY  (X⊗X + Y⊗Y = 2(S⁺S⁻ + S⁻S⁺)).
                γ = -1 →  X⊗X − Y⊗Y = 2(S⁺S⁺ + S⁻S⁻).
    """

    def __init__(self, n_sites: int, n_legs: int,
                 J: float = 1.0, h: float = 0.5, gamma: float = 0.0):
        super().__init__(n_legs, n_phys_sites=n_sites)
        assert n_sites >= 1,       "Need at least one physical site."
        assert n_legs  >= n_sites, "n_legs must be >= n_sites."

        sx, sz, _ = paulis()
        sp, sm    = raising_lowering()

        # Longitudinal field
        for i in range(n_sites):
            self.add_single_site(i, -h * sz)

        # All-to-all XY interaction (real decomposition)
        # X⊗X + γY⊗Y = (1-γ)(S⁺S⁺ + S⁻S⁻) + (1+γ)(S⁺S⁻ + S⁻S⁺)
        prefactor = J / n_sites
        c_diag  = (1 - gamma) * prefactor   # coefficient for S⁺S⁺, S⁻S⁻
        c_cross = (1 + gamma) * prefactor   # coefficient for S⁺S⁻, S⁻S⁺

        for i in range(n_sites):
            for j in range(i + 1, n_sites):
                if c_cross != 0.0:
                    self.add_two_site(i, j, -c_cross * sp, sm)  # S⁺_i S⁻_j
                    self.add_two_site(i, j, -c_cross * sm, sp)  # S⁻_i S⁺_j
                if c_diag != 0.0:
                    self.add_two_site(i, j, -c_diag * sp, sp)   # S⁺_i S⁺_j
                    self.add_two_site(i, j, -c_diag * sm, sm)   # S⁻_i S⁻_j

