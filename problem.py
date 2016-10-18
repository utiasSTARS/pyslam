import copy
import itertools

import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg

from liegroups import SE3


class Problem:
    """Class for building optimization problems."""

    def __init__(self):
        self.residual_blocks = []
        self.param_list = []
        self.block_param_ids = []

    def add_residual_block(self, block, params):
        self.residual_blocks.append(block)

        param_ids = []
        for p in params:
            if p not in self.param_list:
                self.param_list.append(p)
            param_ids.append(self.param_list.index(p))

        self.block_param_ids.append(param_ids)

    def solve(self):
        update_ranges = []
        for p in self.param_list:
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

            dx = self.solve_one_iter()
            # print("Update vector:\n", str(dx))
            print("Update norm = %f" % np.linalg.norm(dx))

            # Backtrack line search
            # step_size = 1
            # best_step_size = step_size
            # best_cost = np.inf
            # alpha = 0.5
            # search_done = False
            #
            # while not search_done:
            #     test_params = copy.deepcopy(self.param_list)
            #
            #     for p, r in zip(test_params, update_ranges):
            #         p.perturb(step_size * dx[r])
            #
            #     test_cost = self.eval_cost(test_params)
            #     print(step_size, " : ", test_cost)
            #     if test_cost < best_cost:
            #         best_cost = test_cost
            #         best_step_size = step_size
            #     else:
            #         search_done = True
            #
            #     step_size = alpha * step_size
            #
            # print("Best step size: %f" % best_step_size)
            # print("Best cost: %f" % best_cost)

            best_step_size = 0.5  # TODO: WHY?

            # Final update
            for p, r in zip(self.param_list, update_ranges):
                print("Before:\n", str(p))
                p.perturb(best_step_size * dx[r])
                print("After:\n", str(p))

            cost = self.eval_cost()
            print("Cost: %f --> %f\n\n" % (prev_cost, cost))

    def solve_one_iter(self):
        b_blocks = [[None] for _ in range(len(self.residual_blocks))]
        A_blocks = [[None for _ in range(len(self.param_list))]
                    for _ in range(len(self.residual_blocks))]

        block_ridx = 0
        for block, pids in zip(self.residual_blocks, self.block_param_ids):
            params = [self.param_list[pid] for pid in pids]
            residual, jacobians = block.evaluate(params, True)

            for pid, jac in zip(pids, jacobians):
                jac_times_weight = jac.T.dot(block.weight)

                block_cidx = pid

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

        return splinalg.spsolve(A, b)

    def eval_cost(self, param_list=None):
        if param_list is None:
            param_list = self.param_list

        cost = 0
        for block, pids in zip(self.residual_blocks, self.block_param_ids):
            params = [param_list[pid] for pid in pids]
            residual = block.evaluate(params)
            cost += residual.dot(block.weight.dot(residual))

        return 0.5 * cost
