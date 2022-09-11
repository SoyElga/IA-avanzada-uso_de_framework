# IA-avanzada-uso_de_framework
Implementación de un modelo de Machine Learning usando un framework

# Descarga
Para el buen uso de este programa se tiene que descargar el paquete de graphviz, puede encontrar la versión de su sistema operativo en la siguiente liga:
https://graphviz.gitlab.io/_pages/Download/Download_windows.html

Una vez descargados los paquetes, al correr el programa le va a pedir que ingrese el path a la dirección de la carpeta bin de Program Files en donde se encuentre Graphviz, ejemplo:
"C:/Program Files/Graphviz/bin/"

Una vez clonado el repositorio y habiendo descargado lo antes mencionado ya se puede usar el programa

# Datos
La base de datos usada es la de Iris de la página de Kaggle, se puede encontrar en este link: https://www.kaggle.com/datasets/uciml/iris
Las variables de la base de datos son las siguientes:
- SepalLengthCm
- SepalWidthCm
- PetalLengthCm
- PetalWidthCm
- Species

Lo que se busca es predecir la especie basandose en las otras variables, en total son 3 especies:
- Iris-setosa
- Iris-virginica
- Iris-versicolor

Para más información de las variables y de la base de datos consulte la página antes mencionada

# Uso
Para usar este programa solo tiene que correr la siguiente linea en una terminal en la que tenga abierta la carpeta del proyecto:
python3 use_decision_tree.py

El programa le dará un menú de opciones para las diferentes cosas que quiera hacer, como predecir datos (estos ya estan cargados en el programa), ver el árbol de decisión o ver las métricas de las predicciones.

# Nota!
El programa esta pensado para que en un futuro se implemente la opción de agregar tu propio set de predicción o set de entrenamiento, actualmente no lo hace pero tampoco te va a pedir información extra sobre estas futuras implementaciones.