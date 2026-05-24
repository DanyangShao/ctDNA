import numpy as np
import pandas as pd
import random
import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras.layers import Input, Dense, Layer
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# =========================================================
# Reproducibility
# =========================================================
seed = 42

np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)

# =========================================================
# Load ctDNA dataset
# =========================================================
data_path = "ctDNA.csv"

df = pd.read_csv(data_path, index_col=0)

print("Loaded full dataset shape:", df.shape)

# =========================================================
# Label columns
# =========================================================
label_cols = [
    "MCC",
    "PRAD",
    "CRC",
    "NSCLC",
    "HCC",
    "BRCA",
    "Normal"
]

# =========================================================
# Feature columns
# =========================================================
feature_cols = [c for c in df.columns if c not in label_cols]

# =========================================================
# Input matrices
# =========================================================
X_all = df[feature_cols].astype(np.float32).values
y_all = df[label_cols].astype(np.float32).values

print("Final X_all:", X_all.shape)
print("Final y_all:", y_all.shape)

# =========================================================
# KL Divergence Layer
# =========================================================
class KLDivergenceLayer(Layer):
    def __init__(self, beta=1.0, **kwargs):
        super(KLDivergenceLayer, self).__init__(**kwargs)
        self.beta = beta

    def call(self, inputs):
        z_mean, z_log_var = inputs

        kl_loss = -0.5 * self.beta * K.mean(
            1 + z_log_var - K.square(z_mean) - K.exp(z_log_var),
            axis=-1,
        )

        self.add_loss(kl_loss)

        return inputs

# =========================================================
# Sampling Layer
# =========================================================
class Sampling(Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs

        eps = tf.random.normal(shape=tf.shape(z_mean))

        return z_mean + tf.exp(0.5 * z_log_var) * eps

# =========================================================
# Build VAE + Classifier
# =========================================================
def build_vae_classifier(input_dim, latent_dim=100, num_classes=7):

    # -----------------------------------------------------
    # Encoder
    # -----------------------------------------------------
    encoder_input = Input(shape=(input_dim,), name="Encoder_input")

    h = Dense(1000, activation="relu", name="enc_dense1")(encoder_input)
    h = Dense(500, activation="relu", name="enc_dense2")(h)

    z_mean = Dense(latent_dim, name="z_mean")(h)
    z_log_var = Dense(latent_dim, name="z_log_var")(h)

    KLDivergenceLayer(beta=1.0, name="kl_loss")([z_mean, z_log_var])

    z = Sampling(name="z")([z_mean, z_log_var])

    encoder = Model(
        encoder_input,
        [z_mean, z_log_var, z],
        name="Encoder"
    )

    # -----------------------------------------------------
    # Decoder
    # -----------------------------------------------------
    decoder_input = Input(shape=(latent_dim,), name="Decoder_input")

    d = Dense(500, activation="relu", name="dec_dense1")(decoder_input)
    d = Dense(1000, activation="relu", name="dec_dense2")(d)

    decoder_output = Dense(
        input_dim,
        activation="sigmoid",
        name="Decoder"
    )(d)

    decoder = Model(
        decoder_input,
        decoder_output,
        name="Decoder"
    )

    # -----------------------------------------------------
    # Classifier
    # -----------------------------------------------------
    classifier_input = Input(shape=(latent_dim,), name="Classifier_input")

    c = Dense(100, activation="relu", name="clf_dense1")(classifier_input)

    classifier_output = Dense(
        num_classes,
        activation="softmax",
        name="Classifier"
    )(c)

    classifier = Model(
        classifier_input,
        classifier_output,
        name="Classifier"
    )

    # -----------------------------------------------------
    # Full VAE model
    # -----------------------------------------------------
    z_out = encoder(encoder_input)[2]

    full_output_dec = decoder(z_out)
    full_output_clf = classifier(z_out)

    full_model = Model(
        encoder_input,
        [full_output_dec, full_output_clf],
        name="VAE_with_classifier"
    )

    return full_model, encoder, decoder, classifier

# =========================================================
# Prepare labels
# =========================================================
y_all_idx = np.argmax(y_all, axis=1)

# =========================================================
# 5-fold cross validation
# =========================================================
skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

fold_conf = []
fold_results = []
fold_latent = []

results = {
    "accuracy": [],
    "precision": [],
    "recall": [],
    "f1": []
}

fold = 1

# =========================================================
# Cross-validation loop
# =========================================================
for trainval_idx, test_idx in skf.split(X_all, y_all_idx):

    print(f"\n=================== FOLD {fold} ===================")

    # -----------------------------------------------------
    # Outer split
    # -----------------------------------------------------
    X_trainval = X_all[trainval_idx]
    y_trainval = y_all[trainval_idx]

    idx_trainval = y_all_idx[trainval_idx]

    X_test = X_all[test_idx]
    y_test = y_all[test_idx]

    idx_test = y_all_idx[test_idx]

    # -----------------------------------------------------
    # Inner split
    # 60 / 20 / 20
    # -----------------------------------------------------
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=0.25,
        random_state=42,
        stratify=idx_trainval
    )

    print(
        "Train:", X_train.shape,
        "Val:", X_val.shape,
        "Test:", X_test.shape
    )

    # -----------------------------------------------------
    # Build model
    # -----------------------------------------------------
    input_dim = X_all.shape[1]

    model, encoder, decoder, classifier = build_vae_classifier(
        input_dim=input_dim,
        latent_dim=100,
        num_classes=7
    )

    # -----------------------------------------------------
    # Load pretrained CancerNet weights
    # -----------------------------------------------------
    pretrained = "34_class_CancerNet_weights.h5"

    model.load_weights(
        pretrained,
        by_name=True,
        skip_mismatch=True
    )

    # =====================================================
    # Stage 1: Freeze encoder layers
    # =====================================================
    for layer in encoder.layers[:3]:
        layer.trainable = False

    model.compile(
        optimizer=Adam(1e-4),
        loss={
            "Decoder": "mse",
            "Classifier": "categorical_crossentropy"
        },
        metrics={
            "Classifier": "accuracy"
        }
    )

    early_stop_stage1 = EarlyStopping(
        monitor="val_loss",
        patience=20,
        restore_best_weights=True
    )

    model.fit(
        X_train,
        {
            "Decoder": X_train,
            "Classifier": y_train
        },
        validation_data=(
            X_val,
            {
                "Decoder": X_val,
                "Classifier": y_val
            }
        ),
        epochs=50,
        batch_size=32,
        callbacks=[early_stop_stage1],
        verbose=0
    )

    # =====================================================
    # Stage 2: Fine-tune full network
    # =====================================================
    for layer in model.layers:
        layer.trainable = True

    model.compile(
        optimizer=Adam(1e-5),
        loss={
            "Decoder": "mse",
            "Classifier": "categorical_crossentropy"
        },
        metrics={
            "Classifier": "accuracy"
        }
    )

    early_stop_stage2 = EarlyStopping(
        monitor="val_loss",
        patience=20,
        restore_best_weights=True
    )

    model.fit(
        X_train,
        {
            "Decoder": X_train,
            "Classifier": y_train
        },
        validation_data=(
            X_val,
            {
                "Decoder": X_val,
                "Classifier": y_val
            }
        ),
        epochs=70,
        batch_size=32,
        callbacks=[early_stop_stage2],
        verbose=0
    )

    # =====================================================
    # Testing
    # =====================================================
    z_mean, z_log, z_latent = encoder.predict(X_test)

    y_pred_prob = classifier.predict(z_latent)

    y_pred = np.argmax(y_pred_prob, axis=1)

    y_true = idx_test

    # =====================================================
    # Confusion matrix
    # =====================================================
    fold_conf.append(
        confusion_matrix(y_true, y_pred)
    )

    fold_results.append((y_true, y_pred))

    fold_latent.append(z_latent)

    # =====================================================
    # Metrics
    # =====================================================
    acc = accuracy_score(y_true, y_pred)

    prec = precision_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    rec = recall_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    print(f"FOLD {fold} RESULTS:")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1-score:  {f1:.4f}")
    print("------------------------------------")

    results["accuracy"].append(acc)
    results["precision"].append(prec)
    results["recall"].append(rec)
    results["f1"].append(f1)

    # =====================================================
    # Classification report
    # =====================================================
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=label_cols,
            zero_division=0
        )
    )

    fold += 1

# =========================================================
# Final summary
# =========================================================
print("\n====== FINAL RESULTS ======")

print("Accuracy :", np.mean(results["accuracy"]))
print("Precision:", np.mean(results["precision"]))
print("Recall   :", np.mean(results["recall"]))
print("F1-score :", np.mean(results["f1"]))

print("\nRaw fold results:")
print(results)
