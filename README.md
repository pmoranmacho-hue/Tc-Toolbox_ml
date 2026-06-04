# Toolbox ML
## 📌 Descripción del Proyecto

Librería Python para análisis exploratorio de datos (EDA), selección de variables y visualización para proyectos de Machine Learning.

---

## Instalación

```bash
git clone https://github.com/pmoranmacho-hue/Tc-Toolbox_ml.git
cd Tc-Toolbox_ml
pip install -r requirements.txt
pip install -e .
```
---

## Funcionalidades

- Descripción de DataFrames.
- Tipificación automática de variables.
- Selección de variables numéricas para regresión.
- Selección de variables categóricas para regresión.
- Visualización de relaciones entre variables y target.

---

## 📁 Estructura del Repositorio

``` Tc-Toolbox_ml/
│
├── notebooks/
│   └── demo.ipynb
│
├── tests/
│   ├── __init__.py
│   └── test_core.py
│
├── toolbox_ml/
│   ├── eda/
│   │    ├── __init__.py
│   │    └── core.py
│   │ 
│   └── __init__.py
│
├── .gitignore
├── README.md
├── requirements.txt
└── .setup.py
```
---

## 🛠️ Tecnologías Utilizadas

- **Python**
- **Pandas** — manipulación y análisis de datos
- **NumPy** — operaciones numéricas
- **Matplotlib** — visualizaciones base
- **Seaborn** — visualizaciones estadísticas
- **SciPy** — entorno de desarrollo


---

## 👥 Equipo y División del Trabajo

| Persona | Rol | Responsabilidades |
|---------|-----|-------------------|
| Pablo Morán | Scrum Master | Setup del repo, setup.py, __init__.py, integración final, notebook demo |
| Ana Belén Escobar | Desarrollador 1 | describe_df + tipifica_variables + sus tests |
| Lucía Vetrano| Desarollador 2 | get/plot_features_num_regression + sus test |
| | Desarrollador 3 | get/plot_features_cat_regression + sus tests + función bonus |

---
