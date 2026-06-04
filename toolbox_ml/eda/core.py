import pandas as pd
from scipy import stats

def get_features_num_regression(
    df: pd.DataFrame,
    target_col: str,
    umbral_corr: float,
    pvalue: float = None
) -> list:
    """
    Devuelve columnas numéricas con correlación significativa con el target.

    Argumentos:
        df (pd.DataFrame): DataFrame a analizar.
        target_col (str): Nombre de la columna target numérica.
        umbral_corr (float): Umbral mínimo de correlación de Pearson (0-1).
        pvalue (float): Nivel de significancia estadística. Opcional.

    Lo que devuelve:
        list: Lista de columnas que superan el umbral. None si falla validación.
    """
    # Validaciones de entrada
    if not isinstance(df, pd.DataFrame):
        print("Error: df debe ser un pd.DataFrame.")
        return None
    if target_col not in df.columns:
        print(f"Error: '{target_col}' no existe en el DataFrame.")
        return None
    if not pd.api.types.is_numeric_dtype(df[target_col]):
        print("Error: target_col debe ser una columna numérica.")
        return None
    if not isinstance(umbral_corr, (int, float)) or not (0 <= umbral_corr <= 1):
        print("Error: umbral_corr debe ser un float entre 0 y 1.")
        return None
    if pvalue is not None and not (0 <= pvalue <= 1):
        print("Error: pvalue debe ser un float entre 0 y 1.")
        return None

    # Seleccionamos columnas numéricas excepto el target
    columnas_num = df.select_dtypes(include='number').columns.tolist()
    columnas_num = [c for c in columnas_num if c != target_col]

    resultado = []

    for col in columnas_num:
        # Eliminamos nulos para el cálculo
        datos = df[[col, target_col]].dropna()
        corr, p = stats.pearsonr(datos[col], datos[target_col])

        # Filtramos por umbral de correlación
        if abs(corr) >= umbral_corr:
            # Si se especifica pvalue, filtramos también por significancia
            if pvalue is None or p < pvalue:
                resultado.append(col)

    return resultado


def plot_features_num_regression(
    df: pd.DataFrame,
    target_col: str = "",
    columns: list = [],
    umbral_corr: float = 0,
    pvalue: float = None
) -> list:
    """
    Pinta pairplots de columnas numéricas correlacionadas con el target.

    Argumentos:
        df (pd.DataFrame): DataFrame a analizar.
        target_col (str): Nombre de la columna target.
        columns (list): Columnas candidatas. Si está vacía usa todas las numéricas.
        umbral_corr (float): Umbral mínimo de correlación (0-1).
        pvalue (float): Nivel de significancia. Opcional.

    Lo que devuelve:
        list: Columnas representadas. None si falla validación.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Reutilizamos las validaciones de get_features_num_regression
    if not isinstance(df, pd.DataFrame):
        print("Error: df debe ser un pd.DataFrame.")
        return None
    if target_col not in df.columns:
        print(f"Error: '{target_col}' no existe en el DataFrame.")
        return None
    if not pd.api.types.is_numeric_dtype(df[target_col]):
        print("Error: target_col debe ser numérica.")
        return None
    if not isinstance(umbral_corr, (int, float)) or not (0 <= umbral_corr <= 1):
        print("Error: umbral_corr debe ser un float entre 0 y 1.")
        return None

    # Si columns está vacía usamos todas las numéricas
    if not columns:
        columns = df.select_dtypes(include='number').columns.tolist()
        columns = [c for c in columns if c != target_col]

    # Filtramos por correlación
    columnas_validas = []
    for col in columns:
        datos = df[[col, target_col]].dropna()
        corr, p = stats.pearsonr(datos[col], datos[target_col])
        if abs(corr) >= umbral_corr:
            if pvalue is None or p < pvalue:
                columnas_validas.append(col)

    if not columnas_validas:
        print("No hay columnas que superen los criterios.")
        return []

    # Si hay más de 5 columnas las dividimos en grupos de 5
    grupos = [columnas_validas[i:i+4] for i in range(0, len(columnas_validas), 4)]

    for grupo in grupos:
        cols_plot = [target_col] + grupo
        sns.pairplot(df[cols_plot].dropna())
        plt.show()

    return columnas_validas