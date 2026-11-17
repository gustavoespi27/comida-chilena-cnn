"""Prueba rápida: Transfer Learning (MobileNetV2, EfficientNetB0) vs. CNN experimental con más épocas, en GPU.

Uso:  python scripts/exportar_split_nb05.py split_nb05.npz
      python scripts/probar_transfer_learning.py split_nb05.npz
"""
import os, sys, time, json
os.environ["KERAS_BACKEND"]="torch"
import numpy as np, keras
from keras import layers
d=np.load(sys.argv[1]); x_tr,y_tr,x_va,y_va=d["x_train"].astype("float32"),d["y_train"],d["x_val"].astype("float32"),d["y_val"]
def aum():
    return [layers.RandomFlip("horizontal"),layers.RandomRotation(0.10),layers.RandomZoom(0.10)]
def transfer(nombre):
    if nombre=="mobilenetv2":
        base=keras.applications.MobileNetV2(input_shape=(224,224,3),include_top=False,weights="imagenet"); pre=keras.applications.mobilenet_v2.preprocess_input
    else:
        base=keras.applications.EfficientNetB0(input_shape=(224,224,3),include_top=False,weights="imagenet"); pre=keras.applications.efficientnet.preprocess_input
    base.trainable=False
    i=layers.Input((224,224,3)); x=i
    for a in aum(): x=a(x)
    x=pre(x); x=base(x,training=False); x=layers.GlobalAveragePooling2D()(x); x=layers.Dropout(0.3)(x)
    return keras.Model(i,layers.Dense(8,activation="softmax")(x)), base
def correr(nombre, m, lr, epocas, base=None, ft_epocas=0):
    keras.utils.set_random_seed(42)
    m.compile(keras.optimizers.Adam(lr),"sparse_categorical_crossentropy",metrics=["accuracy"])
    cb=[keras.callbacks.EarlyStopping("val_loss",patience=8,restore_best_weights=True),keras.callbacks.ReduceLROnPlateau("val_loss",factor=0.5,patience=3,min_lr=1e-6)]
    t=time.time(); h=m.fit(x_tr,y_tr,validation_data=(x_va,y_va),epochs=epocas,batch_size=32,callbacks=cb,verbose=0)
    vl,va=m.evaluate(x_va,y_va,verbose=0); print(f">>> {nombre}: val_loss={vl:.4f} val_acc={va:.4f} épocas={len(h.history['loss'])} {time.time()-t:.0f}s",flush=True)
    if base is not None and ft_epocas:
        base.trainable=True
        for l in base.layers[:-30]: l.trainable=False
        for l in base.layers:
            if isinstance(l,layers.BatchNormalization): l.trainable=False
        m.compile(keras.optimizers.Adam(1e-5),"sparse_categorical_crossentropy",metrics=["accuracy"])
        t=time.time(); h=m.fit(x_tr,y_tr,validation_data=(x_va,y_va),epochs=ft_epocas,batch_size=32,callbacks=cb,verbose=0)
        vl,va=m.evaluate(x_va,y_va,verbose=0); print(f">>> {nombre}+finetune: val_loss={vl:.4f} val_acc={va:.4f} épocas={len(h.history['loss'])} {time.time()-t:.0f}s",flush=True)
for n in ["mobilenetv2","efficientnetb0"]:
    m,b=transfer(n); correr(n,m,1e-3,25,b,15)
# CNN experimental del NB05 desde cero, más épocas
m=keras.Sequential([layers.Input((224,224,3)),*aum(),layers.Rescaling(1/255)]+[l for f in (32,64,128,256) for l in (layers.Conv2D(f,3,activation="relu",padding="same"),layers.MaxPooling2D())]+[layers.GlobalAveragePooling2D(),layers.Dense(128,activation="relu"),layers.Dropout(0.4),layers.Dense(8,activation="softmax")])
correr("cnn_experimental_60ep",m,1e-3,60)
