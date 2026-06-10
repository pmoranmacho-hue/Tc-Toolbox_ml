<<<<<<< Updated upstream
=======
import pandas as pd
from scipy import stats
from toolbox_ml.eda.core import get_features_num_regression, plot_features_num_regression, get_features_cat_regression, plot_features_cat_regression, detect_outliers

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

def test_get_features_cat_regression_devuelve_lista():
    """Caso correcto: input válido → retorna lista."""
    df = pd.DataFrame({
        'cat': ['A', 'A', 'B', 'B', 'B', 'A'],
        'target': [10, 11, 30, 31, 29, 12]
    })

    resultado = get_features_cat_regression(df, 'target', pvalue=0.05)

    assert isinstance(resultado, list)


def test_get_features_cat_regression_detecta_categoria_significativa():
    """Variable categórica con grupos claramente distintos debe aparecer."""
    df = pd.DataFrame({
        'cat': ['A'] * 10 + ['B'] * 10,
        'target': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2,
                   100, 101, 100, 101, 100, 101, 100, 101, 100, 101]
    })

    resultado = get_features_cat_regression(df, 'target', pvalue=0.05)

    assert 'cat' in resultado


def test_get_features_cat_regression_ignora_categoria_no_significativa():
    """Variable categórica sin diferencias claras no debe aparecer."""
    df = pd.DataFrame({
        'cat': ['A', 'A', 'A', 'B', 'B', 'B'],
        'target': [1, 2, 3, 1, 2, 3]
    })

    resultado = get_features_cat_regression(df, 'target', pvalue=0.05)

    assert 'cat' not in resultado


def test_get_features_cat_regression_retorna_none_input_invalido():
    """Caso de error: inputs inválidos → retorna None."""
    assert get_features_cat_regression("no es df", 'target', 0.05) is None
    assert get_features_cat_regression(pd.DataFrame({'a': [1]}), 'no_existe', 0.05) is None
    assert get_features_cat_regression(
        pd.DataFrame({'cat': ['A'], 'target': ['no_num']}),
        'target',
        0.05
    ) is None
    assert get_features_cat_regression(
        pd.DataFrame({'cat': ['A'], 'target': [1]}),
        'target',
        1.5
    ) is None


def test_plot_features_cat_regression_devuelve_lista():
    """Caso correcto: input válido → retorna lista."""
    df = pd.DataFrame({
        'cat': ['A'] * 10 + ['B'] * 10,
        'target': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2,
                   100, 101, 100, 101, 100, 101, 100, 101, 100, 101]
    })

    resultado = plot_features_cat_regression(df, 'target', pvalue=0.05)

    assert isinstance(resultado, list)


def test_plot_features_cat_regression_usa_columns():
    """Si se pasan columnas concretas, solo evalúa esas columnas."""
    df = pd.DataFrame({
        'cat_buena': ['A'] * 10 + ['B'] * 10,
        'cat_mala': ['X', 'Y'] * 10,
        'target': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2,
                   100, 101, 100, 101, 100, 101, 100, 101, 100, 101]
    })

    resultado = plot_features_cat_regression(
        df,
        target_col='target',
        columns=['cat_buena'],
        pvalue=0.05
    )

    assert 'cat_buena' in resultado
    assert 'cat_mala' not in resultado


def test_plot_features_cat_regression_retorna_none_input_invalido():
    """Caso de error: inputs inválidos → retorna None."""
    assert plot_features_cat_regression("no es df", 'target') is None
    assert plot_features_cat_regression(pd.DataFrame({'a': [1]}), 'no_existe') is None
    assert plot_features_cat_regression(
        pd.DataFrame({'cat': ['A'], 'target': ['no_num']}),
        'target'
    ) is None
    assert plot_features_cat_regression(
        pd.DataFrame({'cat': ['A'], 'target': [1]}),
        'target',
        pvalue=1.5
    ) is None


def test_detect_outliers_devuelve_diccionario():
    """Caso correcto: retorna diccionario."""
    df = pd.DataFrame({
        'a': [1, 2, 3, 4, 100]
    })

    resultado = detect_outliers(df, columns=['a'])

    assert isinstance(resultado, dict)


def test_detect_outliers_detecta_outlier_iqr():
    """Detecta outlier usando método IQR."""
    df = pd.DataFrame({
        'a': [1, 2, 3, 4, 100]
    })

    resultado = detect_outliers(df, columns=['a'], method='iqr')

    assert 'a' in resultado
    assert 4 in resultado['a']


def test_detect_outliers_retorna_none_input_invalido():
    """Caso de error: inputs inválidos → retorna None."""
    assert detect_outliers("no es df") is None
    assert detect_outliers(pd.DataFrame({'a': [1, 2, 3]}), method='mal') is None
    assert detect_outliers(pd.DataFrame({'a': [1, 2, 3]}), threshold=-1) is None
>>>>>>> Stashed changes
