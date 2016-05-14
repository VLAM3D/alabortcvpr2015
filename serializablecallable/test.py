import tempfile
from nose.tools import raises
import pickle

from serializablecallable.base import extract_source
from serializablecallable import SerializableCallable


path = tempfile.mkstemp()[1]


def test_serialize_callable_and_test_basic():
    from serializablecallable.base import serialize_callable_and_test as sct

    def a_mock_function(x, y):
        pass

    sct(a_mock_function, [])


def test_get_source():

    def a_mock_function(x, y):
        pass

    source = """def a_mock_function(x, y):\n    pass"""
    assert source == extract_source(a_mock_function)


def test_use_allowed_global():
    from serializablecallable.base import serialize_callable_and_test as sct
    from itertools import product
    import itertools

    def a_mock_function(*args):
        return product(args)

    sct(a_mock_function, [itertools])


@raises(NameError)
def test_use_unallowed_global_raises_never_present():
    from serializablecallable.base import serialize_callable_and_test as sct

    def a_mock_function(*args):
        return product(args)

    sct(a_mock_function, [])


@raises(NameError)
def test_use_unallowed_global_raises_was_present():
    from serializablecallable.base import serialize_callable_and_test as sct

    from itertools import product

    def a_mock_function(*args):
        return product(args)

    sct(a_mock_function, [])


def test_save_serializable_callable():
    from itertools import product
    import itertools

    def a_mock_function(*args):
        return product(args)

    sc = SerializableCallable(a_mock_function, [itertools])
    pickle.dumps(sc, protocol=2)


def test_load_serializable_callable():

    def a_function(*args):
        from itertools import product
        return list(product(args))

    sc = SerializableCallable(a_function, [])
    x = pickle.dumps(sc, protocol=2)
    f = pickle.loads(x).callable
    assert a_function(2, 4) == f(2, 4)
