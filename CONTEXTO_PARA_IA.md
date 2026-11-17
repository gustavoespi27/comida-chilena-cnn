# Contexto del proyecto (para un asistente de IA)

> Este archivo resume todo lo necesario para ayudar con este proyecto sin haber visto las conversaciones anteriores.
> Repositorio público: https://github.com/gustavoespi27/comida-chilena-cnn

## 1. Qué es

Trabajo de un estudiante del curso *Taller Fundamentos Data Science Python — Machine Learning 2026* (INACAP).
Las guías originales del profesor están en
https://github.com/juliopez/Taller-Fundamentos-Data-Science-Python/tree/main/Machine_Learning_2026/03_Notebooks.
La tarea pedía aplicar dos de esas guías a un **dataset propio de imágenes**:

- **NB03** — preparación y gestión del dataset (exploración, calidad, división, normalización, *Data Augmentation*).
- **NB05** — construcción y entrenamiento de una CNN desde cero, comparación con una segunda arquitectura y elección de modelo.

El tema elegido es **clasificación de 8 platos típicos chilenos**. Todo el material (textos, comentarios, respuestas) está en **español**.

## 2. Dataset

Carpeta `dataset_comida_chilena/`, una subcarpeta por clase, archivos `<clase>_0001.jpg`, `<clase>_0002.jpg`, …
1.074 imágenes JPEG RGB, tamaños originales variados (ancho 250–6.000 px, alto 150–4.624 px, mediana 1.024 × 768).

| Índice | Clase | Imágenes |
|---|---|---|
| 0 | cazuela | 154 |
| 1 | chacarero | 121 |
| 2 | completo | 124 |
| 3 | empanada | 155 |
| 4 | humitas | 132 |
| 5 | mote_con_huesillo | 92 |
| 6 | pastel_de_choclo | 141 |
| 7 | sopaipilla | 155 |

- El índice de cada clase es el **orden alfabético** de las carpetas (así lo asigna Keras).
- Desbalance moderado: 155 / 92 = 1,68.
- Pares de clases que se parecen: humitas ↔ pastel de choclo (choclo amarillo), chacarero ↔ completo (sándwiches), empanada ↔ sopaipilla (masas doradas).
- Origen: búsquedas de imágenes en la web → eliminación de casi-duplicados con *hash* perceptual → revisión visual automática y manual → renumeración. `fuentes_dataset_comida_chilena.csv` guarda la URL de origen de cada foto. Uso académico.

## 3. Archivos del repositorio

| Archivo | Para qué sirve |
|---|---|
| `NB03_Preparacion_Dataset_Comida_Chilena.ipynb` | Guía NB03 adaptada + sección 17 con evidencias (17.1) y respuestas de la actividad (17.2). |
| `NB05_CNN_Comida_Chilena.ipynb` | Guía NB05 adaptada + sección 23 con análisis (23.1–23.4) y respuestas (23.5). |
| `scripts/generar_notebooks.py` | **Genera los dos notebooks** a partir de las guías originales del profesor (las descarga si no se le pasa carpeta). Reemplaza textos y celdas, y agrega las secciones 17 y 23. |
| `scripts/respuestas/NB03.md`, `NB05.md` | Texto de las respuestas que el generador inserta en 17.2 y 23.5. |
| `scripts/busqueda_gpu.py`, `exportar_split_nb05.py`, `probar_transfer_learning.py` | Pruebas de hiperparámetros y de Transfer Learning en GPU (ver sección 6). |
| `scripts/descargar_dataset.py`, `limpiar_y_revisar.py`, `aplicar_revision.py`, `renumerar.py` | Construcción del dataset (ya se ejecutaron; no hace falta repetirlos). |
| `fuentes_dataset_comida_chilena.csv` | Origen de cada imagen. |

**Importante:** los notebooks se generan con `python scripts/generar_notebooks.py`. Si se edita un `.ipynb` a mano y luego se vuelve a ejecutar el generador, esos cambios se pierden. Para un cambio permanente, edite el generador o los archivos de `scripts/respuestas/`. Para un cambio puntual en Colab, editar el notebook directamente está bien.

## 4. Cómo se ejecuta

- **Colab (lo normal):** botón "Abrir en Colab" del README. La primera celda de datos hace `git clone` del repositorio. El NB05 monta Google Drive solo para guardar modelos en `MyDrive/MachineLearning2026/`. Para el NB05 hay que activar la GPU T4.
- **PC:** `git clone`, `pip install tensorflow scikit-learn pandas matplotlib pillow jupyter`, abrir los notebooks desde la carpeta del repositorio. Sin Colab, los modelos se guardan en `modelos/`. En Windows nativo TensorFlow no usa la GPU (una época del modelo inicial tarda ~14 s en CPU de 12 núcleos).

## 5. Detalles técnicos

**NB03:** `train_test_split` estratificado 70/15/15, semilla 42 → 751 / 161 / 162 imágenes. Pipeline `tf.data`, `resize` a 224 × 224, `Rescaling(1./255)`. Aumento solo en train: `RandomFlip("horizontal")`, `RandomRotation(0.10)`, `RandomZoom(0.10)` (sin flip vertical ni cambios de color, porque el color distingue platos).

**NB05:** `image_dataset_from_directory` con `validation_split=0.20`, `seed=42`, 224 × 224, batch 32 → 860 train / 214 validación.

- **CNN inicial** (110.792 parámetros): aumento → Rescaling → [Conv2D 32 → MaxPool] → [Conv2D 64 → MaxPool] → [Conv2D 128 → MaxPool] → GlobalAveragePooling → Dense 128 → Dropout 0,30 → Dense 8 softmax.
- **CNN experimental** (422.344 parámetros): igual + bloque Conv2D 256 → MaxPool, Dropout 0,40.
- Adam lr 0,001, `sparse_categorical_crossentropy`, 20 épocas.
- La celda 23.4 elige el modelo con mayor *validation accuracy* (empate → menos parámetros) y guarda `cnn_comida_chilena_elegido_v1.keras` + `.json`. El modelo inicial se guarda como `cnn_comida_chilena_inicial_v1.keras`.
- Las respuestas 4, 5, 6, 8 y 9 de `scripts/respuestas/NB05.md` tienen espacios **[ … ]** que el estudiante completa con los valores de su propia ejecución en Colab.

## 6. Resultados de las pruebas de hiperparámetros

Pruebas hechas en una RTX 4060 (Keras 3 con motor PyTorch, porque TensorFlow no usa GPU en Windows), con la misma división 80/20 y semilla 42 del NB05:

| Modelo | val_accuracy | val_loss |
|---|---|---|
| CNN inicial del NB05 (20 épocas) | ≈ 0,52 | ≈ 1,32 |
| CNN experimental, hasta 60 épocas + *early stopping* + ReduceLROnPlateau | 0,594 | 1,165 |
| MobileNetV2, receta exacta del NB06 (10 + 10 épocas, fine-tuning desde la capa 100) | 0,967 | 0,095 |
| **MobileNetV2 ajustada (25 + 15 épocas, early stopping, ReduceLROnPlateau, fine-tuning de las últimas 30 capas)** | **0,981** | **0,063** |
| EfficientNetB0 + fine-tuning (solo prueba; no se usa porque el curso usa MobileNetV2) | 0,995 | 0,026 |

- Una búsqueda automática de hiperparámetros para CNN desde cero (bloques, filtros, BatchNorm, dropout, lr, batch, *weight decay*, aumento; `scripts/busqueda_gpu.py`) se detuvo por tiempo: en 6 épocas ninguna combinación superó a los modelos del NB05. Con 860 imágenes, una CNN desde cero se queda cerca del 60 %.
- Por eso el NB05 tiene una **sección 24** con *Transfer Learning* usando **MobileNetV2, la misma red de la guía NB06 del profesor** (25 épocas con la base congelada + 15 de *fine-tuning* de las últimas 30 capas con lr 1e-5, `EarlyStopping` y `ReduceLROnPlateau`). Guarda `mobilenetv2_comida_chilena_v1.keras` + `.json`. Las secciones 1–23 siguen siendo la entrega pedida por la guía; la 24 es una mejora que adelanta el NB06.
- Scripts: `scripts/exportar_split_nb05.py` (crea el `.npz` con la división) y `scripts/probar_transfer_learning.py`. Para usar la GPU en Windows: `pip install torch --index-url https://download.pytorch.org/whl/cu128` y `KERAS_BACKEND=torch` (ya lo fijan los scripts).

## 7. Cómo ayudar

- Responder en español, con lenguaje simple: el estudiante está aprendiendo.
- **Usar solo las tecnologías del curso:** TensorFlow/Keras, scikit-learn, pandas, numpy, matplotlib y Google Colab; para Transfer Learning, MobileNetV2 (como en el NB06). PyTorch solo se usó en los scripts locales de prueba, nunca en los notebooks.
- No cambiar los nombres de carpetas, clases, archivos de modelo ni la división de datos (semilla 42) sin avisar: las respuestas ya escritas dependen de ellos.
- Si piden algo nuevo (por ejemplo, la guía NB06 de *Transfer Learning*), seguir el mismo estilo: copiar la guía original del profesor, adaptar textos y rutas al dataset de comida chilena, usar la celda que clona el repositorio y agregar al final una sección con evidencias y respuestas a la actividad.
- Si el usuario pega resultados de Colab, usarlos para completar los **[ … ]** de las respuestas.
