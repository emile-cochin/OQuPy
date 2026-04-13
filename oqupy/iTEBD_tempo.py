# Implementation of iTEBD-TEMPO
# Author: Valentin Link (valentin.link@tu-dresden.de)
# Original code from https://github.com/val-link/iTEBD-TEMPO.git
# Modified by Paul Eastham (easthamp@tcd.ie) so that the iTEBD-TEMPO class uses the OQuPy BathCorrelations class 
# to define the bath correlations rather than the bath correlation function itself. 
# Modified by Émile Cochin so that the iTEBD step is optimized for large Hilbert space sizes

from typing import Text

import numpy as np
from scipy.integrate import dblquad
from scipy.linalg import expm, norm, svd
from scipy.sparse.linalg import eigs
from typing import Callable, Optional
from oqupy.util import get_progress
from ncon import ncon  # ncon performs better than np.einsum

from oqupy.bath_correlations import BaseCorrelations


class iTEBD_TEMPO():
    """ A class to compute and approximate the influence functional unsing iTEBD-TEMPO and compute dynamics. """

    def __init__(self, s_vals: np.ndarray, delta: float, bath_correlations: BaseCorrelations, n_c: int, eta: Optional[np.ndarray] = None, progress_type: Optional[Text] = None):
        """
        :param s_vals: Real eigenvalues of the system-bath coupling operator.
        :param delta: time step for Trotter splitting.
        :param bcf: Bath correlation function.
        :param n_c: Memory cutoff. Should be chosen large enough.
        """

        self.progress_type = progress_type
        self.n_c = n_c
        self.n_c_eff = n_c
        self.s_vals = s_vals
        self.s_dim = self.s_vals.size
        self.nu_dim = self.s_vals.size ** 2 + 1
        self.bcf = bath_correlations
        self.delta = delta
        if eta is None:
            self.eta = self.compute_eta(self.n_c, delta, self.progress_type)
        elif len(eta) != self.n_c:
            raise ValueError("Eta values are not conform to dkmax")
        else:
            self.eta = eta
        self.s_diff = np.empty((self.nu_dim - 1), dtype=np.complex128)
        self.s_sum = np.empty((self.nu_dim - 1), dtype=np.complex128)
        for nu in range(self.nu_dim - 1):
            i, j = int(nu / self.s_dim), nu % self.s_dim
            self.s_diff[nu] = self.s_vals[i] - self.s_vals[j]
            self.s_sum[nu] = self.s_vals[i] + self.s_vals[j]
        self.s_diff = np.pad(self.s_diff, [(0, 1)])
        self.s_sum = np.pad(self.s_sum, [(0, 1)])
        self.kron_delta = np.identity(self.nu_dim)
        self.f = None
        return

    def compute_eta(self, n: int, delta: float, progress_type: Optional[Text] = None) -> np.ndarray:
        """
        Computes the discretized bath correlation function (eta) for n time steps delta.
        :param n: Number of time steps.
        :param delta: Time step.
        :return: Discretized bath correlation function.
        """

        prog_bar = get_progress(progress_type)(n, '--> Computing eta values:')
        prog_bar.enter()

        eta = np.zeros(n, dtype=np.complex128)
        eta[0] = self.bcf.correlation_2d_integral(delta,0.0,shape='upper-triangle')
        prog_bar.update(1) 
        for k in range(1, n):
            eta[k] = self.bcf.correlation_2d_integral(delta,k*delta)
            prog_bar.update(k+1)
        prog_bar.exit()
        return eta

    def compute_f(self, rtol: float, rank: Optional[int] = np.inf, show_info=True):
        """
        Compute the infinite influence functional tensor f using the regular iTEBD algorithm.

        :param rtol: Relative tolerance for svd compression.
        :param rank: Maximum allowed rank (bond dimension).
        """
        info_list = []
        A = np.ones((1, self.nu_dim, 1))
        B = np.ones((1, self.nu_dim, 1))
        sAB = np.ones((1))
        sBA = np.ones((1))
        rank_is_one = True

        progress = get_progress(self.progress_type)
        with progress(self.n_c, '--> Building influence functional:') as prog_bar:
            for k in range(1, self.n_c + 1):
                i_tens = np.exp(-self.eta[self.n_c - k].real * np.outer(self.s_diff, self.s_diff) - 1j * self.eta[self.n_c - k].imag * np.outer(self.s_sum, self.s_diff))

                if k % 2 == 0:
                    B, sBA, A, sAB, info = iTEBD_apply_gate1(i_tens, (k == self.n_c), B, sBA, A, sAB, rank, rtol=rtol)
                else:
                    A, sAB, B, sBA, info = iTEBD_apply_gate1(i_tens, (k == self.n_c), A, sAB, B, sBA, rank, rtol=rtol)

                if show_info:
                    info_str = f' d1:{info[0]}/{self.nu_dim}, d2:{info[1]}/{self.nu_dim}, chi:{info[2]}'
                    prog_bar.info = info_str
                    info_list.append(info)

                if rank_is_one:
                    if np.all([sAB.shape[0] == 1, sAB.shape[-1] == 1, sBA.shape[0] == 1, sBA.shape[-1] == 1]):
                        # reset to initial mps if rank is still one
                        sAB = np.ones((1))
                        sBA = np.ones((1))
                        A = np.ones((1, self.nu_dim, 1))
                        B = np.ones((1, self.nu_dim, 1))
                    else:
                        rank_is_one = False
                        self.n_c_eff = self.n_c - k + 1
                        if k == 1:
                            print('Warning: the memory cutoff n_c may be too small for the given rtol value. The algorithm may become unstable and inaccurate. It is recommended to increase n_c until this message does no longer appear.')
                prog_bar.update(k)
        self.f = np.squeeze(ncon([np.diag(sAB), B, np.diag(sBA), A], [[-1, 1], [1, -2, 2], [2, 3], [3, -3, -4]]))
#         self.f = np.squeeze(np.einsum('j,jkl,l,lno->jkno', sAB, B, sBA, A)) 

        if sAB.shape[0] == 1:
            # handle trivial f
            self.f = np.ones((1, self.nu_dim, 1))

        # compute f[:,-1,:]^\inf = v_r * v_l^T using Lanczos
        w, v_r = eigs(self.f[:, -1, :], 1, which='LR')
        w, v_l = eigs(self.f[:, -1, :].T, 1, which='LR')
        self.v_r = v_r[:, 0]
        self.v_l = v_l[:, 0] / (v_l[:, 0] @ v_r[:, 0])

        print('rank ', self.f.shape[0])
        return info_list[:-1]

    def compute_f_enhanced(self, rtol: float, rtol2, rtol3, rank: Optional[int] = np.inf, show_info=True):
        """
        Compute the infinite influence functional tensor f using the enhanced iTEBD.

        :param rtol: Relative tolerance for svd compression.
        :param rank: Maximum allowed rank (bond dimension).
        """
        info_list = []
        A = np.ones((1, self.nu_dim, 1))
        B = np.ones((1, self.nu_dim, 1))
        sAB = np.ones((1))
        sBA = np.ones((1))
        preA, preB = None, None
        rank_is_one = True

        progress = get_progress(self.progress_type)
        with progress(self.n_c, '--> Building influence functional:') as prog_bar:
            for k in range(1, self.n_c + 1):
                i_tens = np.exp(-self.eta[self.n_c - k].real * np.outer(self.s_diff, self.s_diff) - 1j * self.eta[self.n_c - k].imag * np.outer(self.s_sum, self.s_diff))
                if k % 2 == 0:
                    B, sBA, A, sAB, preB, preA, info = iTEBD_apply_gate_enhanced(i_tens, (k == self.n_c), B, sBA, A, sAB, preB, preA, rank, rtol=rtol, rtol2=rtol2, rtol3=rtol3)
                else:
                    A, sAB, B, sBA, preA, preB, info = iTEBD_apply_gate_enhanced(i_tens, (k == self.n_c), A, sAB, B, sBA, preA, preB, rank, rtol=rtol, rtol2=rtol2, rtol3=rtol3)

                if show_info:
                    info_str = f' d1:{info[0]}/{self.nu_dim}, d2:{info[1]}/{self.nu_dim}, chi:{info[2]}'
                    prog_bar.info = info_str
                    info_list.append(info)

                if rank_is_one:
                    if np.all([sAB.shape[0] == 1, sAB.shape[-1] == 1, sBA.shape[0] == 1, sBA.shape[-1] == 1]):
                        # reset to initial mps if rank is still one
                        sAB = np.ones((1))
                        sBA = np.ones((1))
                        A = np.ones((1, self.nu_dim, 1))
                        B = np.ones((1, self.nu_dim, 1))
                        preA, preB = None, None
                    else:
                        rank_is_one = False
                        self.n_c_eff = self.n_c - k + 1
                        if k == 1:
                            print('Warning: the memory cutoff n_c may be too small for the given rtol value. The algorithm may become unstable and inaccurate. It is recommended to increase n_c until this message does no longer appear.')
                prog_bar.update(k)
        self.f = np.squeeze(ncon([np.diag(sAB), B, np.diag(sBA), A], [[-1, 1], [1, -2, 2], [2, 3], [3, -3, -4]]))
#         self.f = np.squeeze(np.einsum('j,jkl,l,lno->jkno', sAB, B, sBA, A)) 

        if sAB.shape[0] == 1:
            # handle trivial f
            self.f = np.ones((1, self.nu_dim, 1))

        # compute f[:,-1,:]^\inf = v_r * v_l^T using Lanczos
        w, v_r = eigs(self.f[:, -1, :], 1, which='LR')
        w, v_l = eigs(self.f[:, -1, :].T, 1, which='LR')
        self.v_r = v_r[:, 0]
        self.v_l = v_l[:, 0] / (v_l[:, 0] @ v_r[:, 0])

        print('rank ', self.f.shape[0])
        return info_list[:-1]


def iTEBD_apply_gate(gate: np.ndarray, endbool, A: np.ndarray, sAB: np.ndarray, B: np.ndarray, sBA: np.ndarray, rank: int, rtol: float, ctol: Optional[float] = 1e-13):
    """
    single iTEBD step, scheme adapted from https://www.tensors.net/mps
    :param gate: TEBD gate for A-B link
    :param A: A tensor (left)
    :param sAB: weight for A-B link
    :param B: B tensor (right)
    :param sBA: weight for B-A link
    :param rank: maximum rank in svd compression
    :param rtol: relative error for svd compression
    :param ctol: cutoff for weights sBA which need to be inverted
    :return: new tensors and weights A, sAB, B, sBA
    """

    # renormalize weights
    sAB = sAB * norm(sBA)
    sBA = sBA / norm(sBA)

    # ensure weights are above tolerance (needed for inversion)
    sBA[np.abs(sBA) < ctol] = ctol

    # MPS - gate contraction
    d1 = gate.shape[1]
    d2 = gate.shape[-1]
    rank_BA = sBA.shape[0]
    if endbool:
        d1 = 1
        u, s_vals, v = svd(np.einsum('a,acd,d,dcg,g,i,c->aicg', sBA, A, sAB, B, sBA, np.ones((1)), np.diagonal(gate)).reshape([d1 * rank_BA, d2 * rank_BA]), full_matrices=False)
    else:
        u, s_vals, v = svd(np.einsum('a,acd,d,dfg,g,fc->afcg', sBA, A, sAB, B, sBA, gate).reshape([d1 * rank_BA, d2 * rank_BA]), full_matrices=False)

    # truncate singular values
    if rtol is None:
        rank_new = min(rank, len(s_vals))
    else:
        s_vals_sum = np.cumsum(s_vals) / np.sum(s_vals)
        rank_rtol = np.searchsorted(s_vals_sum, 1 - rtol) + 1
        rank_new = min(rank, len(s_vals), rank_rtol)
    u = u[:, :rank_new].reshape(sBA.shape[0], d1 * rank_new)
    v = v[:rank_new, :].reshape(rank_new * d2, rank_BA)

    # factor out sAB weights from A and B
    A = (np.diag(1 / sBA) @ u).reshape(sBA.shape[0], d1, rank_new)
    B = (v @ np.diag(1 / sBA)).reshape(rank_new, d2, rank_BA)

    # new weights
    sAB = s_vals[:rank_new]

    return A, sAB, B, sBA, (d1, d2, rank_new)

def iTEBD_apply_gate_enhanced(gate: np.ndarray, endbool, A: np.ndarray, sAB: np.ndarray, B: np.ndarray, sBA: np.ndarray, preA, preB, rank: int, rtol: float, rtol2, rtol3, ctol: Optional[float] = 1e-13):
    """
    single iTEBD step, scheme adapted from https://www.tensors.net/mps
    :param gate: TEBD gate for A-B link
    :param A: A tensor (left)
    :param sAB: weight for A-B link
    :param B: B tensor (right)
    :param sBA: weight for B-A link
    :param rank: maximum rank in svd compression
    :param rtol: relative error for svd compression
    :param ctol: cutoff for weights sBA which need to be inverted
    :return: new tensors and weights A, sAB, B, sBA
    """

    # renormalize weights
    sAB = sAB * norm(sBA)
    sBA = sBA / norm(sBA)

    # ensure weights are above tolerance (needed for inversion)
    sBA[np.abs(sBA) < ctol] = ctol

    # MPS - gate contraction
    d1 = gate.shape[1]
    d2 = gate.shape[-1]
    rank_BA = sBA.shape[0]
    if endbool:
        d1 = 1
        if preA is not None and preB is not None:
            A, B = np.einsum('abc,eb->aec', A, preA), np.einsum('abc,eb->aec', B, preB)
        u, s_vals, v = svd(np.einsum('a,acd,d,dcg,g,i,c->aicg', sBA, A, sAB, B, sBA, np.ones((1)), np.diagonal(gate)).reshape([d1 * rank_BA, d2 * rank_BA]), full_matrices=False)
        d1, d2 = d2, d1
        alpha = None
        u1, u2 = None, None
    else:
        # SVD decomposition of b(k) of steps (a) to (b) in Figure 2
        U1, S, U2 = svd(gate, full_matrices=False) # SVD
        # relative cutoff of eigenvalues
        s_vals_sum = np.cumsum(S)/np.sum(S)
        rank_rtol = np.searchsorted(s_vals_sum, 1 - rtol2) + 1
        rank_new = min(len(S), rank_rtol)
        U1, S, U2 = U1[:, :rank_new], S[:rank_new], U2[:rank_new, :]
        alpha = rank_new
        # building the theta^A/B blocks of step (c)
        if preA is not None and preB is not None:
            thetaA, thetaB = np.einsum('a,aed,ce,d,bc->cabd', sBA, A, preA, np.sqrt(sAB), U2), np.einsum('d,deg,fe,g,fb->fdbg', np.sqrt(sAB), B, preB, sBA, U1)
        else:
            thetaA, thetaB = np.einsum('a,acd,d,bc->cabd', sBA, A, np.sqrt(sAB), U2), np.einsum('d,dfg,g,fb->fdbg', np.sqrt(sAB), B, sBA, U1)

        # partial SVD decomposition of the theta^A block in (c) step
        u1, s1, v1 = svd(thetaA.reshape(thetaA.shape[0], -1), full_matrices=False)
        # relative cutoff of eigenvalues
        s1_sum = np.cumsum(s1)/np.sum(s1)
        rank_rtol = np.searchsorted(s1_sum, 1 - rtol3) + 1
        rank_new = min(len(s1), rank_rtol)
        u1, s1, v1 = u1[:, :rank_new], s1[:rank_new], v1[:rank_new, :]
        thetaA = (np.diag(s1) @ v1).reshape((-1, *thetaA.shape[1:]))

        # partial SVD decomposition of the theta^B block in (c) step
        u2, s2, v2 = svd(thetaB.reshape(thetaB.shape[0], -1), full_matrices=False)
        # relative cutoff of eigenvalues
        s2_sum = np.cumsum(s2)/np.sum(s2)
        rank_rtol = np.searchsorted(s2_sum, 1 - rtol3) + 1
        rank_new = min(len(s2), rank_rtol)
        u2, s2, v2 = u2[:, :rank_new], s2[:rank_new], v2[:rank_new, :]
        thetaB = (np.diag(s2) @ v2).reshape((-1, *thetaB.shape[1:]))
        d1, d2 = thetaA.shape[0], thetaB.shape[0]

        # SVD decomposition of the Theta block of (d) step
        u, s_vals, v = svd(np.einsum('cabd,fdbg,b->afcg', thetaA, thetaB, S).reshape([d2 * rank_BA, d1 * rank_BA]), full_matrices=False)

    # SVD truncation for the Theta block to prevent blowup of the bond dimension
    if rtol is None:
        rank_new = min(rank, len(s_vals))
    else:
        s_vals_sum = np.cumsum(s_vals) / np.sum(s_vals)
        rank_rtol = np.searchsorted(s_vals_sum, 1 - rtol) + 1
        rank_new = min(rank, len(s_vals), rank_rtol)
    u = u[:, :rank_new].reshape(sBA.shape[0], d2 * rank_new)
    v = v[:rank_new, :].reshape(rank_new * d1, rank_BA)

    # factor out sAB weights from A and B
    A = (np.diag(1 / sBA) @ u).reshape(sBA.shape[0], d2, rank_new)
    B = (v @ np.diag(1 / sBA)).reshape(rank_new, d1, rank_BA)
    
    # new weights
    sAB = s_vals[:rank_new]

    return A, sAB, B, sBA, u2, u1, (d1, d2, rank_new, alpha)
