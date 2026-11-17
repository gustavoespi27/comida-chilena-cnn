"""Búsqueda automática de hiperparámetros para la CNN del NB05, en GPU (Keras 3 + PyTorch).

Método (successive halving):
  1. Se prueban N combinaciones al azar durante pocas épocas.
  2. Se descartan las peores (mayor val_loss) y las mejores siguen entrenando, por rondas.
  3. Los finalistas se reentrenan desde cero con otras semillas para confirmar que no fue suerte.

Uso:  python scripts/busqueda_gpu.py <split_nb05.npz> <carpeta_resultados>
(el .npz se crea con scripts/exportar_split_nb05.py)
"""
import os
os.environ.setdefault("KERAS_BACKEND", "torch")

import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import keras
if os.environ.get("MIXTA", "0") == "1":
    keras.config.set_dtype_policy("mixed_float16")
from keras import layers

NPZ, SALIDA = Path(sys.argv[1]), Path(sys.argv[2])
SALIDA.mkdir(parents=True, exist_ok=True)
N_INICIAL = int(os.environ.get("N_INICIAL", 30))
RONDAS = [int(e) for e in os.environ.get("RONDAS", "8,24,60").split(",")]  # épocas acumuladas al final de cada ronda
CONSERVAR = 1 / 3             # fracción que pasa a la ronda siguiente
SEMILLAS_CONFIRMACION = [int(s) for s in os.environ.get("SEMILLAS", "7,123").split(",") if s]

d = np.load(NPZ)
x_tr, y_tr = d["x_train"].astype("float32"), d["y_train"]
x_va, y_va = d["x_val"].astype("float32"), d["y_val"]
N_CLASES = len(d["clases"])

AUMENTOS = {
    "suave": dict(rot=0.10, zoom=0.10, trasl=0.0, contraste=0.0),   # el del NB05
    "medio": dict(rot=0.15, zoom=0.20, trasl=0.10, contraste=0.0),
    "fuerte": dict(rot=0.20, zoom=0.25, trasl=0.15, contraste=0.15),
}


def construir(c):
    a = AUMENTOS[c["aumento"]]
    capas = [layers.Input(shape=(224, 224, 3)), layers.RandomFlip("horizontal"),
             layers.RandomRotation(a["rot"]), layers.RandomZoom(a["zoom"])]
    if a["trasl"]:
        capas.append(layers.RandomTranslation(a["trasl"], a["trasl"]))
    if a["contraste"]:
        capas.append(layers.RandomContrast(a["contraste"]))
    capas.append(layers.Rescaling(1. / 255))
    f = c["filtros"]
    for _ in range(c["bloques"]):
        for _ in range(c["convs_por_bloque"]):
            if c["batchnorm"]:
                capas += [layers.Conv2D(f, 3, padding="same", use_bias=False),
                          layers.BatchNormalization(), layers.Activation("relu")]
            else:
                capas.append(layers.Conv2D(f, 3, padding="same", activation="relu"))
        capas.append(layers.MaxPooling2D())
        f = min(f * 2, 512)
    capas += [layers.GlobalAveragePooling2D(), layers.Dense(c["dense"], activation="relu"),
              layers.Dropout(c["dropout"]), layers.Dense(N_CLASES, activation="softmax", dtype="float32")]
    m = keras.Sequential(capas)
    if c["weight_decay"]:
        opt = keras.optimizers.AdamW(c["lr"], weight_decay=c["weight_decay"])
    else:
        opt = keras.optimizers.Adam(c["lr"])
    m.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m


class Seguimiento(keras.callbacks.Callback):
    """Guarda los mejores pesos (menor val_loss) y baja el learning rate a la mitad si se estanca.
    A diferencia de los callbacks de Keras, conserva su estado entre llamadas a fit()."""

    def __init__(self, paciencia_lr=5, min_lr=1e-5):
        super().__init__()
        self.mejor, self.pesos, self.espera = np.inf, None, 0
        self.paciencia_lr, self.min_lr = paciencia_lr, min_lr
        self.historia = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": [], "lr": []}

    def on_epoch_end(self, epoca, logs=None):
        lr = float(keras.ops.convert_to_numpy(self.model.optimizer.learning_rate))
        for k in ("loss", "accuracy", "val_loss", "val_accuracy"):
            self.historia[k].append(round(float(logs[k]), 4))
        self.historia["lr"].append(lr)
        if logs["val_loss"] < self.mejor - 1e-4:
            self.mejor, self.pesos, self.espera = logs["val_loss"], self.model.get_weights(), 0
        else:
            self.espera += 1
            if self.espera >= self.paciencia_lr and lr > self.min_lr:
                self.model.optimizer.learning_rate.assign(max(lr / 2, self.min_lr))
                self.espera = 0

    def resumen(self):
        h = self.historia
        i = int(np.argmin(h["val_loss"]))
        return dict(epocas=len(h["loss"]), mejor_epoca=i + 1, val_loss=h["val_loss"][i],
                    val_acc=h["val_accuracy"][i], train_acc=h["accuracy"][i], train_loss=h["loss"][i])


def config_al_azar(rng):
    return dict(
        bloques=rng.choice([3, 4, 5]),
        filtros=rng.choice([16, 32, 48]),
        convs_por_bloque=rng.choice([1, 2]),
        batchnorm=rng.choice([True, True, False]),
        dense=rng.choice([64, 128, 256]),
        dropout=rng.choice([0.2, 0.3, 0.4, 0.5]),
        lr=round(10 ** rng.uniform(np.log10(2e-4), np.log10(3e-3)), 5),
        weight_decay=rng.choice([0, 0, 1e-4, 1e-3]),
        batch=rng.choice([16, 32, 64]),
        aumento=rng.choice(["suave", "medio", "fuerte"]),
    )


NB05_INICIAL = dict(bloques=3, filtros=32, convs_por_bloque=1, batchnorm=False, dense=128, dropout=0.3,
                    lr=0.001, weight_decay=0, batch=32, aumento="suave")
NB05_EXPERIMENTAL = dict(NB05_INICIAL, bloques=4, dropout=0.4)


def entrenar(modelo, seg, c, hasta):
    desde = seg.historia and len(seg.historia["loss"])
    modelo.fit(x_tr, y_tr, validation_data=(x_va, y_va), batch_size=c["batch"], initial_epoch=desde,
               epochs=hasta, shuffle=True, callbacks=[seg], verbose=0)


def registrar(archivo, fila):
    with open(SALIDA / archivo, "a", encoding="utf-8") as f:
        f.write(json.dumps(fila, ensure_ascii=False) + "\n")


def log(*a):
    print(*a, flush=True)


def main():
    rng = random.Random(2026)
    candidatos = {"nb05_inicial": NB05_INICIAL, "nb05_experimental": NB05_EXPERIMENTAL}
    for i in range(N_INICIAL):
        candidatos[f"c{i:02d}"] = config_al_azar(rng)

    vivos = {}
    for nombre, c in candidatos.items():
        keras.utils.set_random_seed(42)
        vivos[nombre] = (construir(c), Seguimiento())

    # Los dos modelos del NB05 siempre se entrenan completos, como referencia.
    referencias = {"nb05_inicial", "nb05_experimental"}
    for r, hasta in enumerate(RONDAS, 1):
        t0 = time.time()
        for nombre, (m, seg) in vivos.items():
            entrenar(m, seg, candidatos[nombre], hasta)
            s = seg.resumen()
            registrar("rondas.jsonl", dict(ronda=r, nombre=nombre, **candidatos[nombre], **s,
                                           parametros=int(m.count_params())))
            log(f"R{r} {nombre:18s} val_loss={s['val_loss']:.4f} val_acc={s['val_acc']:.3f} "
                f"(época {s['mejor_epoca']}/{s['epocas']})")
        orden = sorted(vivos, key=lambda n: vivos[n][1].mejor)
        log(f"--- Ronda {r} lista en {(time.time() - t0) / 60:.1f} min. Ranking: {orden[:6]}")
        if r < len(RONDAS):
            n_pasan = max(3, round(len([n for n in orden if n not in referencias]) * CONSERVAR))
            pasan = [n for n in orden if n not in referencias][:n_pasan]
            vivos = {n: vivos[n] for n in pasan + sorted(referencias)}

    # Confirmación: los 3 mejores (sin contar referencias) desde cero con otras semillas.
    finalistas = [n for n in orden if n not in referencias][:3] + sorted(referencias)
    for nombre in finalistas:
        for semilla in SEMILLAS_CONFIRMACION:
            keras.utils.set_random_seed(semilla)
            m, seg = construir(candidatos[nombre]), Seguimiento()
            entrenar(m, seg, candidatos[nombre], RONDAS[-1])
            s = seg.resumen()
            registrar("confirmacion.jsonl", dict(nombre=nombre, semilla=semilla, **candidatos[nombre], **s,
                                                 parametros=int(m.count_params()), historia=seg.historia))
            log(f"CONF {nombre:18s} semilla={semilla} val_loss={s['val_loss']:.4f} val_acc={s['val_acc']:.3f}")
    log("FIN")


if __name__ == "__main__":
    main()
