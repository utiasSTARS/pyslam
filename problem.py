import copy
import itertools

import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg


class Options:
    """Class for specifying optimization options."""

    def __init__(self):
        self.max_iters = 100
        self.min_update_norm = 1e-6
        self.min_cost = 1e-12

        self.linesearch_alpha = 0.8
        self.linesearch_max_iters = 10
        self.linesearch_min_cost_decrease = 0.9

        self.allow_nondecreasing_steps = False
        self.max_nondecreasing_steps = 3


class Problem:
    """Class for building optimization problems."""

    def __init__(self, options=Options()):
        self.options = options
        """Optimization options."""

        self.param_list = []
        """List of all parameters to be optimized."""

        self.residual_blocks = []
        """List of residual blocks."""
        self.block_param_ids = []
        """Indices of parameters in self.param_list that each block depends on."""

        self.constant_param_ids = []
        """List of parameters to be held constant."""

    def add_residual_block(self, block, params):
        self.residual_blocks.append(block)

        param_ids = []
        for p in params:
            if p not in self.param_list:
                self.param_list.append(p)
            param_ids.append(self.param_list.index(p))

        self.block_param_ids.append(param_ids)

    def set_parameters_constant(self, params):
        for p in params:
            pid = self.param_list.index(p)
            if pid not in self.constant_param_ids:
                self.constant_param_ids.append(pid)

    def set_parameters_variable(self, params):
        for p in params:
            pid = self.param_list.index(p)
            if pid not in self.constant_param_ids:
                self.constant_param_ids.remove(pid)

    def solve(self):
        variable_params = self._get_params_to_update()
        update_ranges = self._get_update_ranges()

        optimization_iters = 0
        dx = np.array([100])
        prev_cost = np.inf
        cost = np.inf
        nondecreasing_steps = 0

        done_optimization = False

        while not done_optimization:
            optimization_iters += 1

            prev_cost = cost

            if self.options.allow_nondecreasing_steps and \
                    nondecreasing_steps == 0:
                best_params = copy.deepcopy(self.param_list)

            dx = self.solve_one_iter()
            # print("Update vector:\n", str(dx))
            # print("Update norm = %f" % np.linalg.norm(dx))

            # Backtrack line search
            best_step_size = self._do_line_search(dx, update_ranges)

            # Final update
            for p, r in zip(variable_params, update_ranges):
                # print("Before:\n", str(p))
                self._perturb(p, best_step_size * dx[r])
                # print("After:\n", str(p))

            cost = self.eval_cost()

            # Check if done optimizing
            done_optimization = optimization_iters > self.options.max_iters or \
                np.linalg.norm(dx) < self.options.min_update_norm or \
                cost < self.options.min_cost

            if self.options.allow_nondecreasing_steps:
                if cost > prev_cost:
                    nondecreasing_steps += 1
                else:
                    nondecreasing_steps = 0

                if nondecreasing_steps > self.options.max_nondecreasing_steps:
                    done_optimization = True
                    # Careful with rebinding here
                    for p, bp in zip(self.param_list, best_params):
                        p.bindto(bp)
            else:
                done_optimization = done_optimization or cost > prev_cost

            # Status message
            print("iter: %d | Cost: %10e --> %10e" %
                  (optimization_iters, prev_cost, cost))

    def solve_one_iter(self):
        # (H.T * W * H) dx = -H.T * W * e
        H_blocks = [[None for _ in self.param_list]
                    for _ in self.residual_blocks]
        W_diag_blocks = [None for _ in self.residual_blocks]
        e_blocks = [[None] for _ in self.residual_blocks]

        block_ridx = 0
        for block, pids in zip(self.residual_blocks, self.block_param_ids):
            params = [self.param_list[pid] for pid in pids]
            compute_jacobians = [False if pid in self.constant_param_ids
                                 else True for pid in pids]

            # Drop the residual if all the parameters used to compute it are
            # being held constant
            if any(compute_jacobians):
                residual, jacobians = block.evaluate(params, compute_jacobians)

                for pid, jac in zip(pids, jacobians):
                    if jac is not None:
                        block_cidx = pid
                        H_blocks[block_ridx][
                            block_cidx] = sparse.csr_matrix(jac)

                W_diag_blocks[block_ridx] = sparse.csr_matrix(block.weight)
                e_blocks[block_ridx][0] = residual

            block_ridx += 1

        # Annoying cleanup
        W_diag_blocks = [block for block in W_diag_blocks if block is not None]
        e_blocks = [block for block in e_blocks if block[0] is not None]

        H = sparse.bmat(H_blocks, format='csr')
        W = sparse.block_diag(W_diag_blocks, format='csr')
        e = np.bmat(e_blocks).A.T.flatten()

        HW = H.T.dot(W)
        A = HW.dot(H)
        b = -HW.dot(e)

        dx = splinalg.spsolve(A, b)

        return dx

    def solve_one_iter2(self):
        # (H.T * S.T * S * H) dx = -H.T * S.T * e
        SH_blocks = [[None for _ in self.param_list]
                    for _ in self.residual_blocks]
        W_diag_blocks = [None for _ in self.residual_blocks]
        e_blocks = [[None] for _ in self.residual_blocks]

        block_ridx = 0
        for block, pids in zip(self.residual_blocks, self.block_param_ids):
            params = [self.param_list[pid] for pid in pids]
            compute_jacobians = [False if pid in self.constant_param_ids
                                 else True for pid in pids]

            # Drop the residual if all the parameters used to compute it are
            # being held constant
            if any(compute_jacobians):
                residual, jacobians = block.evaluate(params, compute_jacobians)

                for pid, jac in zip(pids, jacobians):
                    if jac is not None:
                        block_cidx = pid
                        H_blocks[block_ridx][
                            block_cidx] = sparse.csr_matrix(jac)

                W_diag_blocks[block_ridx] = sparse.csr_matrix(block.weight)
                e_blocks[block_ridx][0] = residual

            block_ridx += 1

        # Annoying cleanup
        W_diag_blocks = [block for block in W_diag_blocks if block is not None]
        e_blocks = [block for block in e_blocks if block[0] is not None]

        H = sparse.bmat(H_blocks, format='csr')
        W = sparse.block_diag(W_diag_blocks, format='csr')
        e = np.bmat(e_blocks).A.T.flatten()

        HW = H.T.dot(W)
        A = HW.dot(H)
        b = -HW.dot(e)

        dx = splinalg.spsolve(A, b)

        return dx

    def eval_cost(self, param_list=None):
        if param_list is None:
            param_list = self.param_list

        cost = 0
        for block, pids in zip(self.residual_blocks, self.block_param_ids):
            params = [param_list[pid] for pid in pids]
            residual = block.evaluate(params)
            cost += residual.dot(block.weight.dot(residual))

        return 0.5 * cost

    def _get_update_ranges(self):
        update_ranges = []
        for p in self.param_list:
            if self.param_list.index(p) not in self.constant_param_ids:
                if hasattr(p, 'dof'):
                    dof = p.dof
                else:
                    dof = p.size

                if not update_ranges:
                    update_ranges.append(range(dof))
                else:
                    update_ranges.append(range(
                        update_ranges[-1].stop,
                        update_ranges[-1].stop + dof))

        return update_ranges

    def _get_params_to_update(self, param_list=None):
        if param_list is None:
            param_list = self.param_list

        return [p for p in param_list
                if param_list.index(p) not in self.constant_param_ids]

    def _do_line_search(self, dx, update_ranges):
        step_size = 1
        best_step_size = step_size
        best_cost = np.inf

        iters = 0
        done_linesearch = False

        while not done_linesearch:
            iters += 1
            test_params = copy.deepcopy(self.param_list)
            variable_params = self._get_params_to_update(test_params)

            for p, r in zip(variable_params, update_ranges):
                self._perturb(p, step_size * dx[r])

            test_cost = self.eval_cost(test_params)

            # print(step_size, " : ", test_cost)

            if iters < self.options.linesearch_max_iters and \
                    test_cost < \
                    self.options.linesearch_min_cost_decrease * best_cost:
                best_cost = test_cost
                best_step_size = step_size
            else:
                if test_cost < best_cost:
                    best_cost = test_cost
                    best_step_size = step_size

                done_linesearch = True

            step_size = self.options.linesearch_alpha * step_size

        # print("Best step size: %f" % best_step_size)
        # print("Best cost: %f" % best_cost)

        return best_step_size

    def _perturb(self, param, dx):
        if hasattr(param, 'perturb'):
            param.perturb(dx)
        else:
            # Default vector space behaviour
            param += dx

    def _bindto(self, dst, src):
        if hasattr(dst, 'bindto'):
            dst.bindto(src)
        else:
            # Default numpy array behaviour
            dst.data = src.data
