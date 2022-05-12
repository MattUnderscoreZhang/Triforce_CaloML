from addFeatures import safeDivide
import numpy as np


def test_safe_divide():
    assert safeDivide(1.0, 2.0) == 0.5
    assert safeDivide(1.0, 0.0) == 0
    assert safeDivide(0.0, 0.0) == 0
    assert safeDivide(0.0, 2.0) == 0
    assert safeDivide(1.0, 2.5) == 1.0 / 2.5

    num = np.array([1.0, 2.0, 3.0])
    denum = np.array([0.0, 1.0, 2.0])
    assert (safeDivide(num, denum) == np.array([0.0, 2.0, 1.5])).all()
    assert (safeDivide(denum, num) == np.array([0.0, 0.5, 2.0/3.0])).all()

    num = np.array([[1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0]])
    denum = np.array([[0.0, 1.0, 2.0],
                      [4.0, 0.0, 6.0]])
    assert (safeDivide(num, denum) == np.array([[0.0, 2.0, 1.5],
                                                [1.0, 0.0, 1.0]])).all()
    assert (safeDivide(denum, num) == np.array([[0.0, 0.5, 2.0 / 3.0],
                                                [1.0, 0.0, 1.0]])).all()
