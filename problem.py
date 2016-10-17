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
        update_ranges = []
        for p in unique_params:
            if not update_ranges:
                update_ranges.append(range(p.dof))
            else:
                update_ranges.append(range(
                    update_ranges[-1].stop,
                    update_ranges[-1].stop + p.dof))

        max_iters = 10

        num_iters = 0
        dx = np.array([100])
        prev_cost = np.inf
        cost = np.inf
        while num_iters < max_iters and \
                np.linalg.norm(dx) > 1e-2 and \
                cost <= prev_cost:
            prev_cost = cost

            num_iters += 1
            print("iter = %d" % num_iters)

            cost, dx = self.solve_one_iter(unique_params)
            print("Cost: %f" % cost)
            print("Update vector:\n", str(dx))
            print("Norm = %f" % np.linalg.norm(dx))

            for p, r in zip(unique_params, update_ranges):
                print("Before:\n", str(p))
                # TODO: Figure out where this factor of 2 comes from
                p.perturb(0.5 * dx[r])
                print("After:\n", str(p))
            print()

    def solve_one_iter(self, unique_params):
        b_blocks = [[None] for _ in range(len(self.residual_blocks))]
        A_blocks = [[None for _ in range(len(unique_params))]
                    for _ in range(len(self.residual_blocks))]

        cost = 0
        block_ridx = 0
        for block, params in zip(self.residual_blocks, self.block_params):

            residual, jacobians = block.evaluate(params, True)

            for par, jac in zip(params, jacobians):
                jac_times_weight = jac.T.dot(block.weight)
                cost += residual.dot(block.weight.dot(residual))

                block_cidx = unique_params.index(par)

                # Not sure if CSR or CSC is the best choice here.
                # spsolve requires one or the other
                # TODO: Check timings for both
                A_blocks[block_ridx][block_cidx] = sparse.csr_matrix(
                    jac_times_weight.dot(jac))

                if b_blocks[block_ridx][0] is None:
                    b_blocks[block_ridx][0] = sparse.csr_matrix(
                        jac_times_weight.dot(residual)).T
                else:
                    b_blocks[block_ridx][0] += sparse.csr_matrix(
                        jac_times_weight.dot(residual)).T

            block_ridx += 1

        A = sparse.bmat(A_blocks, format='csr')
        b = sparse.bmat(b_blocks, format='csr')

        dx = splinalg.spsolve(A, b)

        return 0.5 * cost, dx

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
