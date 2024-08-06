import numpy as np
from numpy.random import RandomState
from pytest import fixture

from pearl.sample import draw_from_trunc_norm


@fixture
def random_state():
    return RandomState(seed=42)

@fixture
def expected_n_equals_5():
    return np.array([44.88371519161826, 51.76484945753248, 48.15980272793935, 46.87206819764374, 42.47176203913114])

def test_draw_from_trunc_norm_positive_n(random_state, expected_n_equals_5):
    '''
    It should return the same values when passing the same random seed.
    '''
    
    result = draw_from_trunc_norm(a=18, b=85, mu=46, sigma=3.49, n=5, random_state=random_state)
    
    assert np.allclose(result, expected_n_equals_5)
    

def test_draw_from_trunc_norm_0_n(random_state):
    '''
    It should return an empty list if n is 0.
    '''
    result = draw_from_trunc_norm(a=18, b=85, mu=46, sigma=3.49, n=0, random_state=random_state)
    
    assert np.allclose(result, np.array([]))
