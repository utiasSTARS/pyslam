import numpy as np

from pyslam.problem import Problem


def test_residual_blocks():
    from pyslam.costs import QuadraticCost
    problem = Problem()
    param_keys = ['a', 'b', 'c']
    problem.add_residual_block(QuadraticCost(2., 4., 1.),
                               param_keys)
    assert(
        param_keys == problem.block_param_keys[0]
    )


def test_param_dict():
    problem = Problem()
    params = {'a': 1, 'b': 2, 'c': 3}
    problem.initialize_params(params)
    assert(
        problem.param_dict == params
    )
    extra_param = {'d': 4}
    params.update(extra_param)
    problem.initialize_params(extra_param)
    assert(
        problem.param_dict == params
    )


def test_constant_params():
    problem = Problem()
    problem.set_parameters_constant('a')
    assert(
        problem.constant_param_keys == ['a']
    )
    problem.set_parameters_constant(['a', 'b_param'])
    assert(
        problem.constant_param_keys == ['a', 'b_param']
    )
    problem.set_parameters_variable('a')
    assert(
        problem.constant_param_keys == ['b_param']
    )
    problem.set_parameters_variable('c')
    assert(
        problem.constant_param_keys == ['b_param']
    )
    problem.set_parameters_variable(['a', 'b_param', 'c'])
    assert(
        problem.constant_param_keys == []
    )


def test_eval_cost():
    from pyslam.costs import QuadraticCost
    problem = Problem()
    good_params = {'a': 1., 'b': 2., 'c': 1.}
    bad_params = {'a': 1., 'b': 0., 'c': 0.}
    cost1 = QuadraticCost(1., 4., 0.5)
    cost2 = QuadraticCost(0., 1., 2.)
    problem.add_residual_block(cost1, ['a', 'b', 'c'])
    problem.add_residual_block(cost2, ['a', 'b', 'c'])
    problem.initialize_params(good_params)
    assert(
        problem.eval_cost() == 0.
        and problem.eval_cost(bad_params) == 0.5 * ((0.5 * 3. * 3.)
                                                    + (2. * 1. * 1.))
    )


def test_solve_quadratic():
    from pyslam.costs import QuadraticCost

    params_true = {'a': 1., 'b': -2., 'c': 3.}
    params_init = {'a': -20., 'b': 10., 'c': -30.}

    x_data = np.linspace(-5, 5, 10)
    y_data = params_true['a'] * x_data * x_data \
        + params_true['b'] * x_data + params_true['c']

    problem = Problem()
    for x, y in zip(x_data, y_data):
        problem.add_residual_block(QuadraticCost(
            x, y, 1.), params_init)

    problem.initialize_params(params_init)
    params_final = problem.solve()

    for p_final, p_true in zip(params_final.values(), params_true.values()):
        assert(np.allclose(p_final, p_true))
