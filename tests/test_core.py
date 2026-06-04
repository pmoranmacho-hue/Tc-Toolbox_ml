import pandas as pd
from scipy import stats
from toolbox_ml.eda.core import get_features_num_regression, plot_features_num_regression

# Tests get_features_num_regression

def test_get_features_num_regression_devuelve_lista():
    """Caso correcto: input válido → retorna lista."""
    df = pd.DataFrame({
        'a': [1, 2, 3, 4, 5],
        'b': [2, 4, 6, 8, 10],
        'target': [1, 2, 3, 4, 5]
    })
    resultado = get_features_num_regression(df, 'target', umbral_corr=0.5)
    assert isinstance(resultado, list)

def test_get_features_num_regression_detecta_correlacion():
    """Columna perfectamente correlacionada debe aparecer en el resultado."""
    df = pd.DataFrame({
        'a': [1, 2, 3, 4, 5],
        'target': [1, 2, 3, 4, 5]
    })
    resultado = get_features_num_regression(df, 'target', umbral_corr=0.9)
    assert 'a' in resultado

def test_get_features_num_regression_filtra_baja_correlacion():
    """Columna sin correlación no debe aparecer en el resultado."""
    df = pd.DataFrame({
        'a': [1, 2, 3, 4, 5],
        'b': [5, 1, 4, 2, 3],
        'target': [1, 2, 3, 4, 5]
    })
    resultado = get_features_num_regression(df, 'target', umbral_corr=0.9)
    assert 'b' not in resultado

def test_get_features_num_regression_retorna_none_input_invalido():
    """Caso de error: inputs inválidos → retorna None."""
    assert get_features_num_regression("no es df", 'target', 0.5) is None
    assert get_features_num_regression(pd.DataFrame({'a': [1]}), 'no_existe', 0.5) is None
    assert get_features_num_regression(pd.DataFrame({'a': [1], 'target': [1]}), 'target', 1.5) is None