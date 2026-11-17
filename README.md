# Clasificación de comida chilena con CNN

Proyecto de clasificación de imágenes con redes neuronales convolucionales (CNN). Se aplican las guías
**NB03** (preparación del dataset) y **NB05** (construcción y entrenamiento de una CNN) del
*Taller Fundamentos Data Science Python — Machine Learning 2026* a un dataset propio de **8 platos típicos chilenos**.

| Notebook | Contenido | Abrir |
|---|---|---|
| **NB03** — Preparación del dataset | Exploración, control de calidad, división train/validation/test, normalización y Data Augmentation | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gustavoespi27/comida-chilena-cnn/blob/main/NB03_Preparacion_Dataset_Comida_Chilena.ipynb) |
| **NB05** — CNN | Entrenamiento de dos arquitecturas, curvas, matriz de confusión, métricas por clase y elección del modelo. Sección extra con *Transfer Learning* (EfficientNetB0): ≈ 99 % de *validation accuracy* | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gustavoespi27/comida-chilena-cnn/blob/main/NB05_CNN_Comida_Chilena.ipynb) |

## Dataset

1.074 imágenes JPEG, organizadas en una carpeta por clase:

| Clase | Imágenes |
|---|---|
| cazuela | 154 |
| chacarero | 121 |
| completo | 124 |
| empanada | 155 |
| humitas | 132 |
| mote_con_huesillo | 92 |
| pastel_de_choclo | 141 |
| sopaipilla | 155 |

**Cómo se construyó:**
1. Descarga desde buscadores de imágenes, con 12 búsquedas distintas por clase ([`scripts/descargar_dataset.py`](scripts/descargar_dataset.py)).
2. Eliminación automática de casi-duplicados (*hash* perceptual) y de miniaturas de menos de 150 px ([`scripts/limpiar_y_revisar.py`](scripts/limpiar_y_revisar.py)).
3. Revisión visual de todas las imágenes: se descartaron ingredientes, preparación, afiches con texto, collages, personas y otros platos ([`scripts/aplicar_revision.py`](scripts/aplicar_revision.py)), y luego hubo una segunda revisión manual.
4. Numeración correlativa por clase ([`scripts/renumerar.py`](scripts/renumerar.py)).

[`fuentes_dataset_comida_chilena.csv`](fuentes_dataset_comida_chilena.csv) registra, para cada imagen, la URL de origen, la página donde se publicó y la búsqueda que la encontró.

> **Derechos de las imágenes:** las fotografías pertenecen a sus respectivos autores y se usan solo con fines académicos. El CSV de fuentes indica el origen de cada una. Si usted es autor de alguna imagen y quiere que se retire, abra un *issue*.

## Resultados

| Modelo | val_accuracy | val_loss |
|---|---|---|
| CNN inicial (NB05) | ≈ 0,52 | ≈ 1,32 |
| CNN experimental, 60 épocas + *early stopping* | ≈ 0,59 | ≈ 1,16 |
| EfficientNetB0 + *fine-tuning* | ≈ 0,995 | ≈ 0,03 |

## Cómo ejecutar

### En Google Colab (recomendado)
1. Abra el notebook con el botón **Abrir en Colab** de la tabla.
2. **Solo para el NB05:** active la GPU en *Entorno de ejecución → Cambiar tipo de entorno de ejecución → T4 GPU*.
3. Ejecute todo con *Entorno de ejecución → Ejecutar todas*.

El notebook descarga este repositorio automáticamente (`git clone`); no hay que subir nada a Drive. El NB05 pide acceso a Google Drive solo para **guardar los modelos entrenados** en `MyDrive/MachineLearning2026/`, porque Colab borra sus archivos al cerrar la sesión.

### En un PC
```bash
git clone https://github.com/gustavoespi27/comida-chilena-cnn.git
cd comida-chilena-cnn
pip install tensorflow scikit-learn pandas matplotlib pillow jupyter
jupyter notebook
```
Fuera de Colab, los notebooks usan la carpeta del repositorio y guardan los modelos en `modelos/`. TensorFlow no usa la GPU en Windows nativo, así que el entrenamiento del NB05 será más lento.

## Estructura

```text
comida-chilena-cnn/
├── NB03_Preparacion_Dataset_Comida_Chilena.ipynb
├── NB05_CNN_Comida_Chilena.ipynb
├── dataset_comida_chilena/          # una carpeta por clase
├── fuentes_dataset_comida_chilena.csv
├── CONTEXTO_PARA_IA.md              # resumen del proyecto para darle contexto a un asistente de IA
└── scripts/                         # dataset, generación de notebooks y pruebas de hiperparámetros
```
