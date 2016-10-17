from liegroups import SE3
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg
import itertools


class Problem:
    """Class for building optimization problems."""

    def __init__(self):
        self.residual_blocks = []
        self.block_params = []

    def add_residual_block(self, block, params):
        self.residual_blocks.append(block)
        self.block_params.append(params)

    def solve(self):
        all_params = [
            i for i in itertools.chain.from_iterable(self.block_params)]
        unique_params = self._get_unique(all_params)

        dx = self.solve_one_iter(unique_params)
        print(dx)

    def solve_one_iter(self, unique_params):
        b_blocks = [[None] for _ in range(len(self.residual_blocks))]
        A_blocks = [[None for _ in range(len(unique_params))]
                    for _ in range(len(self.residual_blocks))]

        block_ridx = 0
        for block, params in zip(self.residual_blocks, self.block_params):

            residual, jacobians = block.evaluate(params, True)

            for par, jac in zip(params, jacobians):
                jac_times_weight = jac.T.dot(block.weight)

                block_cidx = unique_params.index(par)

                A_blocks[block_ridx][block_cidx] = sparse.coo_matrix(
                    jac_times_weight.dot(jac))

                if b_blocks[block_ridx][0] is None:
                    b_blocks[block_ridx][0] = sparse.coo_matrix(
                        jac_times_weight.dot(residual)).T
                else:
                    b_blocks[block_ridx][0] += sparse.coo_matrix(
                        jac_times_weight.dot(residual)).T

            block_ridx += 1

        A = sparse.csr_matrix(sparse.bmat(A_blocks))
        b = sparse.csr_matrix(sparse.bmat(b_blocks))

        return splinalg.spsolve(A, b)

    def _get_unique(self, seq, idfun=None):
        if idfun is None:
            def idfun(x): return x
        seen = {}
        result = []
        for item in seq:
            marker = idfun(item)
            if marker in seen:
                continue
            seen[marker] = 1
            result.append(item)
        return result
