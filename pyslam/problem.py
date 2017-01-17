import copy
# import itertools

import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg


class Options:
    """Class for specifying optimization options."""

    def __init__(self):
        self.max_iters = 100
        """Maximum number of iterations before terminating."""
        self.min_update_norm = 1e-6
        """Minimum update norm before terminating."""
        self.min_cost = 1e-12
        """Minimum cost value before terminating."""

        self.linesearch_alpha = 0.8
        """Factor by which line search step size decreases each iteration."""
        self.linesearch_max_iters = 10
        """Maximum number of line search steps."""
        self.linesearch_min_cost_decrease = 0.9
        """Minimum cost decrease factor to continue the line search."""

        self.allow_nondecreasing_steps = False
        """Enable non-decreasing steps to escape local minima."""
        self.max_nondecreasing_steps = 3
        """Maximum number of non-dereasing steps before terminating."""


class Problem:
    """Class for building optimization problems."""

    def __init__(self, options=Options()):
        self.options = options
        """Optimization options."""

        self.param_dict = dict()
        """Dictionary of all parameters with their current values."""

        self.residual_blocks = []
        """List of residual blocks."""
        self.block_param_keys = []
        """List of parameter keys in param_dict that each block depends on."""

        self.constant_param_keys = []
        """List of parameter keys in param_dict to be held constant."""

    def add_residual_block(self, block, param_keys):
        """Add a cost block to the problem."""
        self.residual_blocks.append(block)
        self.block_param_keys.append(param_keys)

    def initialize_params(self, param_dict):
        """Initialize the parameters in the problem."""
        # update does a shallow copy, which is no good for immutable parameters
        self.param_dict.update(copy.deepcopy(param_dict))

    def set_parameters_constant(self, param_keys):
        """Hold a list of parameters constant."""
        # param_keys must be a list, but don't force the user to create a
        # 1-element list
        if isinstance(param_keys, str):
            param_keys = [param_keys]

        for key in param_keys:
            if key not in self.constant_param_keys:
                self.constant_param_keys.append(key)

    def set_parameters_variable(self, param_keys):
        """Allow a list of parameters to vary."""
        # param_keys must be a list, but don't force the user to create a
        # 1-element list
        if isinstance(param_keys, str):
            param_keys = [param_keys]

        for key in param_keys:
            if key in self.constant_param_keys:
                self.constant_param_keys.remove(key)

    def eval_cost(self, param_dict=None):
        """Evaluate the cost function using given parameter values."""
        if param_dict is None:
            param_dict = self.param_dict

        cost = 0
        for block, keys in zip(self.residual_blocks, self.block_param_keys):
            try:
                params = [param_dict[key] for key in keys]
            except KeyError as e:
                print(
                    "Parameter {} has not been initialized".format(e.args[0]))

            residual = block.evaluate(params)
            cost += np.dot(residual, np.dot(block.weight, residual))

        return 0.5 * cost

    def solve(self):
        """Solve the problem using Gauss - Newton."""
        variable_param_keys = self._get_params_to_update()
        update_ranges = self._get_update_ranges()

        optimization_iters = 0
        dx = np.array([100])
        cost = self.eval_cost()
        nondecreasing_steps_taken = 0

        done_optimization = False

        while not done_optimization:
            optimization_iters += 1

            prev_cost = cost

            if self.options.allow_nondecreasing_steps and \
                    nondecreasing_steps_taken == 0:
                best_params = copy.deepcopy(self.param_dict)

            dx = self.solve_one_iter()
            # print("Update vector:\n", str(dx))
            # print("Update norm = %f" % np.linalg.norm(dx))

            # Backtrack line search
            best_step_size = self._do_line_search(dx, update_ranges)

            # Final update
            for k, r in zip(variable_param_keys, update_ranges):
                # print("Before:\n", str(self.param_dict[k]))
                self._perturb_by_key(k, best_step_size * dx[r])
                # print("After:\n", str(self.param_dict[k]))

            cost = self.eval_cost()

            # Check if done optimizing
            done_optimization = \
                optimization_iters > self.options.max_iters or \
                np.linalg.norm(dx) < self.options.min_update_norm or \
                cost < self.options.min_cost

            if self.options.allow_nondecreasing_steps:
                if cost >= prev_cost:
                    nondecreasing_steps_taken += 1
                else:
                    nondecreasing_steps_taken = 0

                if nondecreasing_steps_taken \
                        >= self.options.max_nondecreasing_steps:
                    done_optimization = True
                    self.param_dict.update(best_params)
            else:
                done_optimization = done_optimization or cost >= prev_cost

            # Status message
            print("iter: %d | Cost: %10e --> %10e" %
                  (optimization_iters, prev_cost, cost))

        return self.param_dict

    def solve_one_iter(self):
        """Solve one iteration of Gauss - Newton."""
        # (H.T * W * H) dx = -H.T * W * e
        H_blocks = [[None for _ in self.param_dict]
                    for _ in self.residual_blocks]
        W_diag_blocks = [None for _ in self.residual_blocks]
        e_blocks = [[None] for _ in self.residual_blocks]

        block_cidx_dict = dict(zip(self.param_dict.keys(),
                                   list(range(len(self.param_dict)))))

        block_ridx = 0
        for block, keys in zip(self.residual_blocks, self.block_param_keys):
            params = [self.param_dict[key] for key in keys]
            compute_jacobians = [False if key in self.constant_param_keys
                                 else True for key in keys]

            # Drop the residual if all the parameters used to compute it are
            # being held constant
            if any(compute_jacobians):
                residual, jacobians = block.evaluate(params, compute_jacobians)

                for key, jac in zip(keys, jacobians):
                    if jac is not None:
                        block_cidx = block_cidx_dict[key]
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

    def _get_params_to_update(self, param_dict=None):
        """Helper function to identify parameters to update."""
        if param_dict is None:
            param_dict = self.param_dict

        return [key for key in self.param_dict.keys()
                if key not in self.constant_param_keys]

    def _get_update_ranges(self):
        """Helper function to partition the full update vector."""
        update_ranges = []
        for key, param in self.param_dict.items():
            if key not in self.constant_param_keys:
                if hasattr(param, 'dof'):
                    # Check if parameter specifies a tangent space
                    dof = param.dof
                elif hasattr(param, '__len__'):
                    # Check if parameter is a vector
                    dof = len(param)
                else:
                    # Must be a scalar
                    dof = 1

                if not update_ranges:
                    update_ranges.append(range(dof))
                else:
                    update_ranges.append(range(
                        update_ranges[-1].stop,
                        update_ranges[-1].stop + dof))

        return update_ranges

    def _do_line_search(self, dx, update_ranges):
        """Backtrack line search to optimize step size in a given direction."""
        step_size = 1
        best_step_size = step_size
        best_cost = np.inf

        iters = 0
        done_linesearch = False

        while not done_linesearch:
            iters += 1
            test_params = copy.deepcopy(self.param_dict)
            variable_param_keys = self._get_params_to_update(test_params)

            for k, r in zip(variable_param_keys, update_ranges):
                self._perturb_by_key(k, best_step_size * dx[r], test_params)

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

    def _perturb_by_key(self, key, dx, param_dict=None):
        """Helper function to update a parameter given an update vector."""
        if not param_dict:
            param_dict = self.param_dict

        if hasattr(param_dict[key], 'perturb'):
            param_dict[key].perturb(dx)
        else:
            # Default vector space behaviour
            param_dict[key] += dx
