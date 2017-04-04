import copy
import time

import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg

from pyslam.losses import L2Loss


class Options:
    """Class for specifying optimization options."""

    def __init__(self):
        self.max_iters = 100
        """Maximum number of iterations before terminating."""
        self.min_update_norm = 1e-6
        """Minimum update norm before terminating."""
        self.min_cost = 1e-12
        """Minimum cost value before terminating."""
        self.min_cost_decrease = 0.9
        """Minimum cost decrease factor to continue optimization."""

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

        self.print_summary = True
        """Print a summary of the optimization after completion."""
        self.print_iter_summary = False
        """Print the initial and final cost after each iteration."""


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
        self.block_loss_functions = []
        """List of loss functions applied to each block. Default: L2Loss."""

        self.constant_param_keys = []
        """List of parameter keys in param_dict to be held constant."""

        self._update_partition_dict = {}
        """Autogenerated list of update vector ranges corresponding to each parameter."""
        self._covariance_matrix = None
        """Covariance matrix of final parameter estimates."""
        self._cost_history = []
        """History of cost values at each iteration of solve."""

    def add_residual_block(self, block, param_keys, loss=L2Loss()):
        """Add a cost block to the problem."""
        # param_keys must be a list, but don't force the user to create a
        # 1-element list
        if isinstance(param_keys, str):
            param_keys = [param_keys]

        self.residual_blocks.append(block)
        self.block_param_keys.append(param_keys)
        self.block_loss_functions.append(loss)

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

        cost = 0.
        for block, keys, loss in zip(self.residual_blocks,
                                     self.block_param_keys,
                                     self.block_loss_functions):
            try:
                params = [param_dict[key] for key in keys]
            except KeyError as e:
                print(
                    "Parameter {} has not been initialized".format(e.args[0]))

            residual = block.evaluate(params)
            cost += np.sum(loss.loss(residual))

        return cost

    def solve(self):
        """Solve the problem using Gauss - Newton."""
        self._update_partition_dict = self._get_update_partition_dict()

        optimization_iters = 0
        dx = np.array([100])
        cost = self.eval_cost()
        nondecreasing_steps_taken = 0

        done_optimization = False
        self._cost_history = [cost]

        while not done_optimization:
            optimization_iters += 1

            prev_cost = self._cost_history[-1]

            if self.options.allow_nondecreasing_steps and \
                    nondecreasing_steps_taken == 0:
                best_params = copy.deepcopy(self.param_dict)

            dx = self.solve_one_iter()
            # print("Update vector:\n", str(dx))
            # print("Update norm = %f" % np.linalg.norm(dx))

            # Update parameters
            for k, r in self._update_partition_dict.items():
                # print("Before:\n", str(self.param_dict[k]))
                self._perturb_by_key(k, dx[r])
                # print("After:\n", str(self.param_dict[k]))

            # Check if done optimizing
            cost = self.eval_cost()

            done_optimization = optimization_iters > self.options.max_iters or \
                np.linalg.norm(dx) < self.options.min_update_norm or \
                cost < self.options.min_cost

            if self.options.allow_nondecreasing_steps:
                if cost >= self.options.min_cost_decrease * prev_cost:
                    nondecreasing_steps_taken += 1
                else:
                    nondecreasing_steps_taken = 0

                if nondecreasing_steps_taken \
                        >= self.options.max_nondecreasing_steps:
                    done_optimization = True
                    self.param_dict.update(best_params)
            else:
                done_optimization = done_optimization or \
                    cost >= self.options.min_cost_decrease * prev_cost

            # Update cost history
            self._cost_history.append(cost)

            # Print iteration summary
            if self.options.print_iter_summary:
                print('Iter: {:3} | Cost: {:12e} --> {:12e}'.format(
                    optimization_iters, prev_cost, cost))

        # Print optimization summary
        if self.options.print_summary:
            print('Iterations: {:3} | Cost: {:12e} --> {:12e}'.format(
                optimization_iters,
                self._cost_history[0], self._cost_history[-1]))

        return self.param_dict

    def solve_one_iter(self):
        """Solve one iteration of Gauss-Newton."""
        # precision * dx = information
        # start = time.perf_counter()
        precision, information = self._build_precision_and_information()
        # end = time.perf_counter()
        # print('build_precision_and_information: {:.5f} s'.format(end - start))

        # start = time.perf_counter()
        dx = splinalg.spsolve(precision, information)
        # end = time.perf_counter()
        # print('spsolve: {:.5f} s'.format(end - start))

        # Backtrack line search
        # start = time.perf_counter()
        best_step_size = self._do_line_search(dx)
        # end = time.perf_counter()
        # print('do_line_search: {:.5f} s'.format(end - start))

        return best_step_size * dx

    def compute_covariance(self):
        """Compute the covariance matrix after solve has terminated."""
        try:
            # Re-evaluate the precision matrix with the final parameters
            precision_matrix, _ = self._build_precision_and_information()
            self._covariance_matrix = splinalg.inv(precision_matrix).toarray()
        except Exception as e:
            print('Covariance computation failed!\n{}'.format(e))

    def get_covariance_block(self, param0, param1):
        """Get the covariance block corresponding to two parameters."""
        try:
            p0_range = self._update_partition_dict[param0]
            p1_range = self._update_partition_dict[param1]
            return np.squeeze(self._covariance_matrix[
                p0_range.start:p0_range.stop, p1_range.start:p1_range.stop])
        except KeyError as e:
            print(
                'Cannot compute covariance for constant parameter {}'.format(e.args[0]))

        return None

    def summary(self):
        """Return a summary of the optimization."""
        if not self._cost_history:
            raise ValueError('solve has not yet been called')

        if format == 'brief':
            # summary = [[summary_header], [format_string.format]]
            pass

        if format == 'full':
            entry_format_string = iter_nums = range(
                len(self._cost_history) - 1)
            summary = []
            for i, ic, fc in zip(iter_nums,
                                 self._cost_history[:-1], self._cost_history[1:]):
                summary.append(entry_format_string.format(i, ic, fc))

            return ''.join(summary)

        raise ValueError('Invalid summary format \'{}\'. '
                         'Options are \'brief\' or \'full\'.'.format(format))

    def _get_update_partition_dict(self):
        """Helper function to partition the full update vector."""
        update_partition_dict = {}
        prev_key = ''
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

                if not update_partition_dict:
                    update_partition_dict.update({key: range(dof)})
                else:
                    update_partition_dict.update({key: range(
                        update_partition_dict[prev_key].stop,
                        update_partition_dict[prev_key].stop + dof)})

                prev_key = key

        return update_partition_dict

    def _build_precision_and_information(self):
        """Helper function to build the precision matrix and information vector for the Gauss - Newton update."""
        # The Gauss-Newton step is given by
        # (H.T * W * H) dx = -H.T * W * e
        # or
        # precision * dx = information
        #
        # However, in our case, W is subsumed into H and e by the stiffness parameter
        # so instead we have
        # (H'.T * H') dx = -H'.T * e'
        # where H' = sqrt(W) * H and e' = sqrt(W) * e
        #
        # Note that this is an exactly equivalent formulation, but avoids needing
        # to explicitly construct and multiply the (possibly very large) W
        # matrix.
        H_blocks = [[None for _ in self.param_dict]
                    for _ in self.residual_blocks]
        e_blocks = [None for _ in self.residual_blocks]

        block_cidx_dict = dict(zip(self.param_dict.keys(),
                                   list(range(len(self.param_dict)))))

        block_ridx = 0
        for block, keys, loss in zip(self.residual_blocks,
                                     self.block_param_keys,
                                     self.block_loss_functions):
            params = [self.param_dict[key] for key in keys]
            compute_jacobians = [False if key in self.constant_param_keys
                                 else True for key in keys]

            # Drop the residual if all the parameters used to compute it are
            # being held constant
            if any(compute_jacobians):
                residual, jacobians = block.evaluate(params, compute_jacobians)
                # Weight for iteratively reweighted least squares
                loss_weight = np.sqrt(loss.weight(residual))

                for key, jac in zip(keys, jacobians):
                    if jac is not None:
                        block_cidx = block_cidx_dict[key]
                        # transposes needed for proper broadcasting
                        H_blocks[block_ridx][block_cidx] = \
                            sparse.csr_matrix((loss_weight.T * jac.T).T)

                e_blocks[block_ridx] = loss_weight * residual

            block_ridx += 1

        H = sparse.bmat(H_blocks, format='csr')
        e = np.squeeze(np.bmat(e_blocks).A)

        precision = H.T.dot(H)
        information = -H.T.dot(e)

        # import ipdb
        # ipdb.set_trace()

        return precision, information

    def _do_line_search(self, dx):
        """Backtrack line search to optimize step size in a given direction."""
        step_size = 1
        best_step_size = step_size
        best_cost = np.inf

        iters = 0
        done_linesearch = False

        while not done_linesearch:
            iters += 1
            test_params = copy.deepcopy(self.param_dict)

            for k, r in self._update_partition_dict.items():
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
