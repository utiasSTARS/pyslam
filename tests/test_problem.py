import numpy as np

from pyslam.problem import Options, Problem


def test_quadratic():
    from pyslam.costs import QuadraticCost

    params_true = [np.array([1., -2., 3.])]
    params_est = [np.array([-20., 10., -30.])]

    x_data = np.linspace(-5, 5, 10)
    y_data = params_true[0][0] * x_data * x_data \
        + params_true[0][1] * x_data + params_true[0][2]

    problem = Problem()
    for x, y in zip(x_data, y_data):
        problem.add_residual_block(QuadraticCost(
            x, y, 1.), params_est)

    problem.solve()

    assert(np.allclose(params_true, params_est))
