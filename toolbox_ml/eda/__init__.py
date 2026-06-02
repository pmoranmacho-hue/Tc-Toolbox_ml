#funciones vacías. Quedaría rellenar.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def describe_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera un resumen estadístico descriptivo de un DataFrame.
    """
    pass

def tipifica_variables(df: pd.DataFrame, umbral_categoria: int, umbral_continua: float) -> pd.DataFrame:
    """
    Sugiere el tipo estadístico de cada variable de un DataFrame.
    """
    pass

def get_features_num_regression(df: pd.DataFrame, target_col: str, umbral_corr: float, pvalue: float = None) -> list:
    """
    Devuelve columnas numéricas con correlación significativa con el target.
    """
    pass

def plot_features_num_regression(df: pd.DataFrame, target_col: str = "", columns: list = [], umbral_corr: float = 0, pvalue: float = None) -> list:
    """
    Pinta pairplots de las columnas numéricas correlacionadas con el target.
    """
    pass

def get_features_cat_regression(df: pd.DataFrame, target_col: str, pvalue: float = 0.05) -> list:
    """
    Devuelve columnas categóricas con relación estadística significativa con el target.
    """
    pass

def plot_features_cat_regression(df: pd.DataFrame, target_col: str = "", columns: list = [], pvalue: float = 0.05, with_individual_plot: bool = False) -> list:
    """
    Pinta histogramas agrupados de las variables categóricas significativas.
    """
    pass