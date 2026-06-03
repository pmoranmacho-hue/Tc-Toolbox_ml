"""
tests/test_core.py  —  Tests para las funciones del Desarrollador 1
=============================================================================
Cubre: describe_df + tipifica_variables

Ejecutar con:
    pytest tests/ -v
"""

import pytest
import pandas as pd
import numpy as np

#Importamos desde el paquete instalado con `pip install -e .`
from toolbox_ml.eda.core import describe_df, tipifica_variables


#Fixtures compartidos

@pytest.fixture
def df_mixto():
    """DataFrame con columnas de distintos tipos y algún nulo."""
    return pd.DataFrame({
        "entero": [1, 2, 3, 4, 5],
        "flotante": [1.1, 2.2, None, 4.4, 5.5],
        "texto": ["a", "b", "b", "c", "c"],
        "booleano": [True, False, True, False, True],
    })


@pytest.fixture
def df_solo_nulos():
    """DataFrame con una columna completamente nula."""
    return pd.DataFrame({"col_nula": [None, None, None]})


#Tests de describe_df

class TestDescribeDf:

    def test_retorna_dataframe_con_input_valido(self, df_mixto):
        """Caso correcto: input válido → retorna pd.DataFrame."""
        resultado = describe_df(df_mixto)
        assert isinstance(resultado, pd.DataFrame)

    def test_columnas_resultado_correctas(self, df_mixto):
        """El DataFrame resultado tiene exactamente las cuatro columnas esperadas."""
        resultado = describe_df(df_mixto)
        assert set(resultado.columns) == {
            "tipo", "porcentaje_nulos", "valores_unicos", "porcentaje_cardinalidad"
        }

    def test_indice_son_nombres_de_columnas(self, df_mixto):
        """El índice del resultado coincide con los nombres de columna del input."""
        resultado = describe_df(df_mixto)
        assert list(resultado.index) == list(df_mixto.columns)

    def test_porcentaje_nulos_correcto(self):
        """Calcula correctamente el porcentaje de nulos (75 %)."""
        df = pd.DataFrame({"a": [1, None, None, None]})
        resultado = describe_df(df)
        assert resultado.loc["a", "porcentaje_nulos"] == pytest.approx(75.0, abs=0.01)

    def test_porcentaje_nulos_sin_nulos(self, df_mixto):
        """Columna sin nulos → porcentaje_nulos == 0.0."""
        resultado = describe_df(df_mixto)
        assert resultado.loc["entero", "porcentaje_nulos"] == 0.0

    def test_valores_unicos_correcto(self, df_mixto):
        """Cuenta correctamente los valores únicos (NaN no se cuenta)."""
        resultado = describe_df(df_mixto)
        #"texto" tiene: "a", "b", "c" --> 3 únicos
        assert resultado.loc["texto", "valores_unicos"] == 3

    def test_porcentaje_cardinalidad_correcto(self):
        """Calcula correctamente el porcentaje de cardinalidad."""
        df = pd.DataFrame({"col": [1, 2, 3, 4]})  #4 únicos / 4 filas = 100 %
        resultado = describe_df(df)
        assert resultado.loc["col", "porcentaje_cardinalidad"] == pytest.approx(100.0, abs=0.01)

    def test_tipo_correcto_para_cada_columna(self, df_mixto):
        """El campo 'tipo' refleja el dtype real de cada columna."""
        resultado = describe_df(df_mixto)
        assert resultado.loc["entero", "tipo"] == df_mixto["entero"].dtype
        assert resultado.loc["texto", "tipo"] == df_mixto["texto"].dtype

    def test_columna_completamente_nula(self, df_solo_nulos):
        """Columna con todos nulos → porcentaje_nulos == 100.0 y valores_unicos == 0."""
        resultado = describe_df(df_solo_nulos)
        assert resultado.loc["col_nula", "porcentaje_nulos"] == pytest.approx(100.0, abs=0.01)
        assert resultado.loc["col_nula", "valores_unicos"] == 0

    def test_dataframe_vacio_retorna_dataframe_sin_filas(self):
        """DataFrame sin columnas → resultado también vacío."""
        resultado = describe_df(pd.DataFrame())
        assert isinstance(resultado, pd.DataFrame)
        assert len(resultado) == 0

    def test_retorna_none_con_string(self):
        """Input de tipo str → retorna None."""
        assert describe_df("esto no es un dataframe") is None

    def test_retorna_none_con_lista(self):
        """Input de tipo lista → retorna None."""
        assert describe_df([1, 2, 3]) is None

    def test_retorna_none_con_none(self):
        """Input None → retorna None."""
        assert describe_df(None) is None


#Tests de tipifica_variables

class TestTipificaVariables:

    @pytest.fixture
    def df_tipifica(self):
        """DataFrame de 100 filas con distintos tipos de variable."""
        np.random.seed(42)
        return pd.DataFrame({
            "binaria":   np.random.choice([0, 1], size=100),
            "categorica": np.random.choice(["A", "B", "C"], size=100),
            "discreta":  np.random.randint(0, 20, size=100),
            "continua":  np.arange(100, dtype=float),   # 100 únicos --> 100 %
        })

    def test_retorna_dataframe(self, df_tipifica):
        """Caso correcto: retorna pd.DataFrame."""
        resultado = tipifica_variables(df_tipifica, umbral_categoria=10, umbral_continua=30.0)
        assert isinstance(resultado, pd.DataFrame)

    def test_columnas_resultado_correctas(self, df_tipifica):
        """El resultado tiene exactamente las columnas 'nombre_variable' y 'tipo_sugerido'."""
        resultado = tipifica_variables(df_tipifica, umbral_categoria=10, umbral_continua=30.0)
        assert list(resultado.columns) == ["nombre_variable", "tipo_sugerido"]

    def test_numero_de_filas(self, df_tipifica):
        """Una fila por columna del DataFrame de entrada."""
        resultado = tipifica_variables(df_tipifica, umbral_categoria=10, umbral_continua=30.0)
        assert len(resultado) == len(df_tipifica.columns)

    def test_detecta_binaria(self, df_tipifica):
        """Columna con cardinalidad 2 --> 'Binaria'."""
        resultado = tipifica_variables(df_tipifica, umbral_categoria=10, umbral_continua=30.0)
        fila = resultado.loc[resultado["nombre_variable"] == "binaria", "tipo_sugerido"].values[0]
        assert fila == "Binaria"

    def test_detecta_categorica(self, df_tipifica):
        """Columna con cardinalidad < umbral_categoria --> 'Categórica'."""
        resultado = tipifica_variables(df_tipifica, umbral_categoria=10, umbral_continua=30.0)
        fila = resultado.loc[resultado["nombre_variable"] == "categorica", "tipo_sugerido"].values[0]
        assert fila == "Categórica"

    def test_detecta_continua(self, df_tipifica):
        """Columna con cardinalidad >= umbral y %card >= umbral_continua --> 'Numérica Continua'."""
        resultado = tipifica_variables(df_tipifica, umbral_categoria=10, umbral_continua=30.0)
        fila = resultado.loc[resultado["nombre_variable"] == "continua", "tipo_sugerido"].values[0]
        assert fila == "Numérica Continua"

    def test_detecta_discreta(self, df_tipifica):
        """Columna con cardinalidad >= umbral y %card < umbral_continua --> 'Numérica Discreta'."""
        resultado = tipifica_variables(df_tipifica, umbral_categoria=10, umbral_continua=30.0)
        fila = resultado.loc[resultado["nombre_variable"] == "discreta", "tipo_sugerido"].values[0]
        assert fila == "Numérica Discreta"

    def test_retorna_none_si_df_invalido(self):
        """df no es DataFrame --> retorna None."""
        assert tipifica_variables("no es df", 10, 30.0) is None

    def test_retorna_none_si_umbral_categoria_float(self, df_tipifica):
        """umbral_categoria float --> retorna None."""
        assert tipifica_variables(df_tipifica, umbral_categoria=5.5, umbral_continua=30.0) is None

    def test_retorna_none_si_umbral_categoria_negativo(self, df_tipifica):
        """umbral_categoria negativo --> retorna None."""
        assert tipifica_variables(df_tipifica, umbral_categoria=-1, umbral_continua=30.0) is None

    def test_retorna_none_si_umbral_continua_mayor_100(self, df_tipifica):
        """umbral_continua > 100 --> retorna None."""
        assert tipifica_variables(df_tipifica, umbral_categoria=10, umbral_continua=110.0) is None

    def test_retorna_none_si_umbral_continua_negativo(self, df_tipifica):
        """umbral_continua < 0 --> retorna None."""
        assert tipifica_variables(df_tipifica, umbral_categoria=10, umbral_continua=-5.0) is None

    def test_dataframe_vacio(self):
        """DataFrame sin columnas --> resultado vacío pero válido."""
        resultado = tipifica_variables(pd.DataFrame(), umbral_categoria=10, umbral_continua=30.0)
        assert isinstance(resultado, pd.DataFrame)
        assert len(resultado) == 0
