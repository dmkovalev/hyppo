import warnings


def test_streamlit_emits_deprecation():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        import importlib, hyppo.streamlit
        importlib.reload(hyppo.streamlit)
    assert any(issubclass(x.category, DeprecationWarning) for x in w)
