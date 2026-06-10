<<<<<<< Updated upstream
=======
"""
Desarrollador 1: describe_df + tipifica_variables
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


#describe_df

def describe_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera un resumen estadístico descriptivo de un DataFrame.

    Para cada columna del DataFrame de entrada devuelve su tipo de dato,
    el porcentaje de valores nulos, el número de valores únicos y el
    porcentaje de cardinalidad respecto al total de filas.

    Argumentos:
        df (pd.DataFrame): DataFrame a analizar.

    Retorna:
        pd.DataFrame: DataFrame con una fila por columna del input y las
        siguientes columnas:
            - 'tipo'                   : tipo de dato de la columna.
            - 'porcentaje_nulos'       : % de NaN sobre el total de filas,
                                         redondeado a 2 decimales.
            - 'valores_unicos'         : número de valores únicos distintos.
            - 'porcentaje_cardinalidad': % de valores únicos sobre el total
                                         de filas, redondeado a 2 decimales.
        Retorna None si el input no es un DataFrame válido.
    """
    #Validación de entrada
    if not isinstance(df, pd.DataFrame):
        print("Error en describe_df: el argumento 'df' debe ser un pd.DataFrame.")
        return None

    total_filas = len(df)

    #Construción de cada métrica como una Series indexada por nombre de columna
    tipos = df.dtypes

    #Porcentaje de nulos: (nulos / total_filas) * 100
    pct_nulos = (df.isnull().sum() / total_filas * 100).round(2)

    #Valores únicos: contamos sin contar NaN como único valor adicional
    n_unicos = df.nunique()

    #Porcentaje de cardinalidad respecto al total de filas
    pct_cardinalidad = (n_unicos / total_filas * 100).round(2)

    #Ensamblamos el DataFrame resultado
    resultado = pd.DataFrame({
        "tipo": tipos,
        "porcentaje_nulos": pct_nulos,
        "valores_unicos": n_unicos,
        "porcentaje_cardinalidad": pct_cardinalidad,
    })

    #El índice ya son los nombres de columna, debido a cómo construimos las Series
    return resultado


#tipifica_variables


def tipifica_variables(
    df: pd.DataFrame,
    umbral_categoria: int,
    umbral_continua: float,
) -> pd.DataFrame:
    """
    Sugiere el tipo estadístico de cada columna de un DataFrame.

    Aplica la siguiente lógica en cascada para cada columna:
        1. Cardinalidad == 2                                      --> "Binaria"
        2. Cardinalidad <  umbral_categoria                       --> "Categórica"
        3. Cardinalidad >= umbral_categoria  Y  %cardinalidad >= umbral_continua
                                                                 --> "Numérica Continua"
        4. Cardinalidad >= umbral_categoria  Y  %cardinalidad <  umbral_continua
                                                                -->  "Numérica Discreta"

    Argumentos:
        df (pd.DataFrame)        : DataFrame a analizar.
        umbral_categoria (int)   : umbral de cardinalidad para distinguir variables
                                   categóricas de numéricas.  Debe ser un entero > 0.
        umbral_continua (float)  : umbral de porcentaje de cardinalidad para distinguir
                                   variables numéricas continuas de discretas.
                                   Debe ser un float entre 0 y 100.

    Retorna:
        pd.DataFrame: DataFrame con columnas 'nombre_variable' y 'tipo_sugerido',
        con una fila por columna del DataFrame de entrada.
        Retorna None si alguna validación de entrada falla.
    """
    #Validaciones de entrada
    if not isinstance(df, pd.DataFrame):
        print("Error en tipifica_variables: 'df' debe ser un pd.DataFrame.")
        return None

    if not isinstance(umbral_categoria, int) or umbral_categoria <= 0:
        print(
            "Error en tipifica_variables: 'umbral_categoria' debe ser un entero positivo."
        )
        return None

    if not isinstance(umbral_continua, (int, float)) or not (0 <= umbral_continua <= 100):
        print(
            "Error en tipifica_variables: 'umbral_continua' debe ser un float entre 0 y 100."
        )
        return None

    total_filas = len(df)
    nombres = []
    tipos_sugeridos = []

    for col in df.columns:
        cardinalidad = df[col].nunique()

        #Porcentaje de cardinalidad respecto al total de filas
        pct_cardinalidad = (cardinalidad / total_filas * 100) if total_filas > 0 else 0

        #Aplicamos la lógica en cascada
        if cardinalidad == 2:
            tipo = "Binaria"
        elif cardinalidad < umbral_categoria:
            tipo = "Categórica"
        elif pct_cardinalidad >= umbral_continua:
            tipo = "Numérica Continua"
        else:
            tipo = "Numérica Discreta"

        nombres.append(col)
        tipos_sugeridos.append(tipo)

    resultado = pd.DataFrame({
        "nombre_variable": nombres,
        "tipo_sugerido": tipos_sugeridos,
    })

    return resultado

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


# =========================
# Desarrollador 3
# =========================

def get_features_cat_regression(
    df: pd.DataFrame,
    target_col: str,
    pvalue: float = 0.05
) -> list:
    """
    Devuelve columnas categóricas con relación estadística significativa
    con un target numérico.

    Para variables categóricas binarias aplica Mann-Whitney U.
    Para variables categóricas con más de dos categorías aplica ANOVA.

    Argumentos:
        df (pd.DataFrame): DataFrame a analizar.
        target_col (str): Nombre de la columna target numérica.
        pvalue (float): Nivel de significancia estadística.

    Lo que devuelve:
        list: Lista de columnas categóricas significativas. None si falla validación.
    """
    if not isinstance(df, pd.DataFrame):
        print("Error: df debe ser un pd.DataFrame.")
        return None
    if target_col not in df.columns:
        print(f"Error: '{target_col}' no existe en el DataFrame.")
        return None
    if not pd.api.types.is_numeric_dtype(df[target_col]):
        print("Error: target_col debe ser una columna numérica.")
        return None
    if not isinstance(pvalue, (int, float)) or not (0 <= pvalue <= 1):
        print("Error: pvalue debe ser un float entre 0 y 1.")
        return None

    columnas_cat = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    columnas_cat = [c for c in columnas_cat if c != target_col]

    resultado = []

    for col in columnas_cat:
        datos = df[[col, target_col]].dropna()
        categorias = datos[col].unique()

        if len(categorias) < 2:
            continue

        grupos = [datos.loc[datos[col] == cat, target_col] for cat in categorias]
        grupos = [grupo for grupo in grupos if len(grupo) > 0]

        try:
            if len(grupos) == 2:
                _, p = stats.mannwhitneyu(
                    grupos[0],
                    grupos[1],
                    alternative="two-sided"
                )
            else:
                _, p = stats.f_oneway(*grupos)
        except Exception:
            continue

        if p < pvalue:
            resultado.append(col)

    return resultado


def plot_features_cat_regression(
    df: pd.DataFrame,
    target_col: str = "",
    columns: list = [],
    pvalue: float = 0.05,
    with_individual_plot: bool = False
) -> list:
    """
    Pinta histogramas del target agrupados por variables categóricas
    significativas respecto a un target numérico.

    Argumentos:
        df (pd.DataFrame): DataFrame a analizar.
        target_col (str): Nombre de la columna target numérica.
        columns (list): Columnas categóricas candidatas. Si está vacía usa todas.
        pvalue (float): Nivel de significancia estadística.
        with_individual_plot (bool): Si True, pinta una figura por variable.

    Lo que devuelve:
        list: Columnas representadas. None si falla validación.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    if not isinstance(df, pd.DataFrame):
        print("Error: df debe ser un pd.DataFrame.")
        return None
    if target_col not in df.columns:
        print(f"Error: '{target_col}' no existe en el DataFrame.")
        return None
    if not pd.api.types.is_numeric_dtype(df[target_col]):
        print("Error: target_col debe ser una columna numérica.")
        return None
    if not isinstance(pvalue, (int, float)) or not (0 <= pvalue <= 1):
        print("Error: pvalue debe ser un float entre 0 y 1.")
        return None
    if columns is not None and not isinstance(columns, list):
        print("Error: columns debe ser una lista.")
        return None

    if not columns:
        columns = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        columns = [c for c in columns if c != target_col]
    else:
        for col in columns:
            if col not in df.columns:
                print(f"Error: '{col}' no existe en el DataFrame.")
                return None
            if pd.api.types.is_numeric_dtype(df[col]):
                print(f"Error: '{col}' debe ser una columna categórica.")
                return None

    columnas_significativas = []

    for col in columns:
        datos = df[[col, target_col]].dropna()
        categorias = datos[col].unique()

        if len(categorias) < 2:
            continue

        grupos = [datos.loc[datos[col] == cat, target_col] for cat in categorias]
        grupos = [grupo for grupo in grupos if len(grupo) > 0]

        try:
            if len(grupos) == 2:
                _, p = stats.mannwhitneyu(
                    grupos[0],
                    grupos[1],
                    alternative="two-sided"
                )
            else:
                _, p = stats.f_oneway(*grupos)
        except Exception:
            continue

        if p < pvalue:
            columnas_significativas.append(col)

    if not columnas_significativas:
        print("No hay columnas categóricas que superen los criterios.")
        return []

    if with_individual_plot:
        for col in columnas_significativas:
            plt.figure(figsize=(8, 4))
            sns.histplot(
                data=df[[col, target_col]].dropna(),
                x=target_col,
                hue=col,
                kde=True
            )
            plt.title(f"Distribución de {target_col} según {col}")
            plt.show()
    else:
        for i in range(0, len(columnas_significativas), 4):
            grupo = columnas_significativas[i:i+4]
            for col in grupo:
                plt.figure(figsize=(8, 4))
                sns.histplot(
                    data=df[[col, target_col]].dropna(),
                    x=target_col,
                    hue=col,
                    kde=True
                )
                plt.title(f"Distribución de {target_col} según {col}")
                plt.show()

    return columnas_significativas


def detect_outliers(
    df: pd.DataFrame,
    columns: list = [],
    method: str = "iqr",
    threshold: float = 1.5
) -> dict:
    """
    Función bonus: detecta outliers en columnas numéricas.

    method='iqr' usa el rango intercuartílico.
    method='zscore' usa puntuaciones z.

    Argumentos:
        df (pd.DataFrame): DataFrame a analizar.
        columns (list): Columnas numéricas a analizar. Si está vacía usa todas.
        method (str): Método de detección, 'iqr' o 'zscore'.
        threshold (float): Umbral para detectar outliers.

    Lo que devuelve:
        dict: Diccionario con columnas como claves e índices de outliers como valores.
    """
    if not isinstance(df, pd.DataFrame):
        print("Error: df debe ser un pd.DataFrame.")
        return None
    if method not in ["iqr", "zscore"]:
        print("Error: method debe ser 'iqr' o 'zscore'.")
        return None
    if not isinstance(threshold, (int, float)) or threshold <= 0:
        print("Error: threshold debe ser un número positivo.")
        return None

    if not columns:
        columns = df.select_dtypes(include="number").columns.tolist()

    outliers = {}

    for col in columns:
        if col not in df.columns:
            print(f"Error: '{col}' no existe en el DataFrame.")
            return None
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue

        serie = df[col].dropna()

        if method == "iqr":
            q1 = serie.quantile(0.25)
            q3 = serie.quantile(0.75)
            iqr = q3 - q1
            limite_inf = q1 - threshold * iqr
            limite_sup = q3 + threshold * iqr
            indices = serie[(serie < limite_inf) | (serie > limite_sup)].index.tolist()
        else:
            zscores = stats.zscore(serie)
            indices = serie[abs(zscores) > threshold].index.tolist()

        outliers[col] = indices

    return outliers
>>>>>>> Stashed changes
