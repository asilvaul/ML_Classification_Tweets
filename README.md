# 📊 Proyecto de Análisis de Sentimientos de Tweets con un modelo de Clasificación.

## 🎯 Problema de Negocio

El objetivo de este proyecto es analizar los sentimientos de los tweets, identificando si los mensajes son positivos o negativos. Utilizando un conjunto de datos de 1,600,000 tweets preprocesados, se pretende generar diversas características que se utilizarán para entrenar un modelo de clasificación.

## ❓ Preguntas Clave

- 📈 **¿Qué tan efectivos son los análisis de sentimientos aplicados sobre datos textuales como los tweets?**
- 🔄 **¿Qué transformaciones y características textuales son relevantes para mejorar la predicción de sentimientos?**
- 😏 **¿Cómo podemos detectar sarcasmo en tweets utilizando técnicas de análisis de sentimientos?**
- 🧮 **¿Qué métricas se pueden utilizar para evaluar el desempeño del modelo?**

## 🚀 Configuración del Ambiente

Asegúrate de tener instaladas las siguientes bibliotecas necesarias para la ejecución del código:

```bash
pip install nltk
pip install emoji
pip install vaderSentiment
pip install textblob
```

### Otras librerías requeridas:

- pandas
- re (Regular Expressions)
- string
- matplotlib
- seaborn
- sklearn

```bash
pip install numpy pandas seaborn matplotlib scikit-learn
```

## 🗄️ Obtención y Tratamiento de Datos

### Cargando la Base de Datos

El conjunto de datos proviene de un archivo CSV llamado `training_1600000_processed_noemoticon.csv`, que contiene 1,600,000 tweets junto con su anotación de sentimiento (0 = negativo, 4 = positivo).

```python
df = pd.read_csv("training_1600000_processed_noemoticon.csv", encoding='latin-1')
df.columns = ['target', 'ids', 'date', 'flag', 'user', 'text']
```

### Tratamiento de Datos

Durante el preprocesamiento se realizan las siguientes operaciones clave:

- 🧹 **Eliminación de URLs, menciones y emojis**: Se remueven del texto ya que pueden agregar ruido al análisis.
- ❌ **Eliminación de signos de puntuación y caracteres especiales**: Para normalizar el texto antes de aplicar las transformaciones.
- 🔡 **Transformación a minúsculas**: Para evitar que las palabras en mayúsculas se traten como diferentes palabras.
- 🛠️ **Generación de características adicionales**: Se generan nuevas variables basadas en el texto, como longitud del tweet, conteo de stopwords, densidad de palabras en mayúsculas, entre otras.

## 📐 Generación de Características

Se generaron varias características a partir del texto de los tweets. Estas incluyen:

1. **Longitud del tweet** (`tweet_length`).
2. **Conteo de emojis** (`emoji_count`).
3. **Conteo de signos de exclamación/interrogación** (`exclamation_count`, `question_count`).
4. **Densidad de palabras en mayúsculas** (`capital_word_density`).
5. **Conteo de palabras** (`word_count`).
6. **Conteo de stopwords** (`stopword_count`).
7. **Conteo de palabras únicas** (`unique_word_count`).
8. **Proporción de palabras repetidas** (`repeated_word_proportion`).
10. **Subjetividad del sentimiento usando TextBlob** (`textblob_subjectivity`).
11. **Conteo de signos de puntuación** (`punctuation_count`).
12. **Conteo de menciones y hashtags** (`mention_count`, `hashtag_count`).
13. **Entropía del texto** (`text_entropy`).
14. **Detección de sarcasmo** usando VADER y análisis del texto (`sarcasm`).


## 🔀 División de los Datos

El dataset fue dividido en conjuntos de entrenamiento, validación y prueba usando una proporción de 70% para entrenamiento, 15% para validación y 15% para prueba.

```python
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
```

## 📊 Visualización

Se generó una visualización que muestra la distribución de las clases (sentimientos) en el conjunto de datos de entrenamiento.

```python
# Crear una figura de tamaño adecuado
plt.figure(figsize=(10, 6))

# Gráfico de barras para ver la distribución del target
ax = sns.countplot(data=df, x='target', palette="pastel", order=sorted(df['target'].unique()))

# Mostrar el gráfico
plt.show()
```

## 🧠 Modelado

Se dividió el conjunto de datos en conjuntos de **entrenamiento**, **prueba** y **validación**, y luego se entrenaron modelos de machine learning (Xgboost y lightgbm) utilizando las características generadas a partir de los textos procesados.

## 📈 Evaluación del Modelo

Finalmente, se evaluó el modelo utilizando métricas de clasificación como **Auc**, **accuracy**, **precision**, **recall**, **F1-score**. Además, se realizaron ajustes adicionales de hiperparámetros mediante técnicas como GridSearchCV y validación cruzada.

## 🏁 Conclusiones

Este proyecto utiliza técnicas avanzadas de preprocesamiento de texto para crear un modelo capaz de predecir el sentimiento de tweets y detectar sarcasmo en ellos. El modelo resultante puede ser mejorado con técnicas adicionales de NLP y ajustando los hiperparámetros según los resultados obtenidos.
