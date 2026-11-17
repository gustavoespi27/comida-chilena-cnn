"""Guarda en un .npz la misma división 80/20 (seed 42, 224x224) que usa el NB05, para entrenar rápido."""
import sys
import numpy as np
import tensorflow as tf
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
salida = Path(sys.argv[1]) if len(sys.argv) > 1 else RAIZ / "split_nb05.npz"
kw = dict(validation_split=0.20, seed=42, image_size=(224, 224), batch_size=64, shuffle=True)
partes = {}
for subset in ("training", "validation"):
    ds = tf.keras.utils.image_dataset_from_directory(str(RAIZ / "dataset_comida_chilena"), subset=subset, **kw)
    x, y = zip(*[(a.numpy(), b.numpy()) for a, b in ds])
    partes[subset] = (np.concatenate(x).round().clip(0, 255).astype("uint8"), np.concatenate(y))
    clases = ds.class_names
np.savez(salida, x_train=partes["training"][0], y_train=partes["training"][1],
         x_val=partes["validation"][0], y_val=partes["validation"][1], clases=np.array(clases))
print(salida, {k: v[0].shape for k, v in partes.items()})
