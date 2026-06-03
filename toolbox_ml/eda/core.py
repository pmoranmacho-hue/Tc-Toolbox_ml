"""
toolbox_ml/eda/core.py
======================
Módulo principal de funciones EDA para el paquete toolbox_ml.

Desarrollador 1: describe_df + tipifica_variables
"""

import pandas as pd
import numpy as np


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
