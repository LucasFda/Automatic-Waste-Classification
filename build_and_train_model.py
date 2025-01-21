import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np

def get_next_index(directory):
    """
    Obtient le prochain index basé sur les fichiers existants dans le dossier donné.

    Args:
        directory (str): Chemin vers le dossier.

    Returns:
        int: Prochain index à utiliser.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        return 1
    files = [f for f in os.listdir(directory) if f.startswith('model_') and f.endswith('.keras')]
    if not files:
        return 1
    indexes = [int(f.split('_')[1].split('.')[0]) for f in files]
    return max(indexes) + 1

def build_model(input_shape=(150, 150, 3), num_classes=4):
    """
    Construit un modèle CNN pour la classification d'images.

    Args:
        input_shape (tuple): La taille des images (hauteur, largeur, canaux).
        num_classes (int): Nombre de classes à prédire.

    Returns:
        tf.keras.Model: Le modèle compilé.
    """
    from tensorflow.keras.regularizers import l2

    model = Sequential([
        Input(shape=input_shape),
        Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Flatten(),
        Dense(512, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer=Adam(learning_rate=1e-4),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def plot_training_history(history, output_dir, index):
    """
    Génère des graphes de perte et d'accuracy pendant l'entraînement.

    Args:
        history (History): Historique de l'entraînement.
        output_dir (str): Dossier où sauvegarder les graphes.
        index (int): Index du modèle courant.
    """
    # Courbe de perte
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Courbe de perte')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'loss_curve_{index}.png'))
    plt.close()

    # Courbe d'accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Courbe d\'accuracy')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'accuracy_curve_{index}.png'))
    plt.close()

def train_model(data_dir, batch_size=16, epochs=35):
    """
    Entraîne un modèle CNN sur un dataset d'images.

    Args:
        data_dir (str): Chemin vers le dossier contenant 'train', 'test', 'val'.
        batch_size (int): Taille des lots pour l'entraînement.
        epochs (int): Nombre d'époques d'entraînement.

    Returns:
        None
    """
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')

    # Augmentation des données
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.2,
        zoom_range=0.3,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    val_test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir, target_size=(150, 150), batch_size=batch_size, class_mode='categorical')

    val_generator = val_test_datagen.flow_from_directory(
        val_dir, target_size=(150, 150), batch_size=batch_size, class_mode='categorical')

    test_generator = val_test_datagen.flow_from_directory(
        test_dir, target_size=(150, 150), batch_size=batch_size, class_mode='categorical', shuffle=False)

    model = build_model(input_shape=(150, 150, 3), num_classes=train_generator.num_classes)

    # Déterminer l'index pour les fichiers
    model_dir = os.path.join(os.getcwd(), 'models')
    index = get_next_index(model_dir)

    # Checkpoints et callbacks
    checkpoint_path = os.path.join(model_dir, f'model_{index}.keras')
    checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True, verbose=1)
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, verbose=1)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

    history = model.fit(train_generator,
                        validation_data=val_generator,
                        epochs=epochs,
                        callbacks=[checkpoint, early_stopping, lr_scheduler])

    # Sauvegarder les courbes
    plot_training_history(history, model_dir, index)

    # Évaluation finale
    test_loss, test_accuracy = model.evaluate(test_generator)
    print(f"\nTest Loss: {test_loss}, Test Accuracy: {test_accuracy}")

    # Génération du rapport de classification
    test_generator.reset()
    predictions = model.predict(test_generator)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = test_generator.classes
    class_labels = list(test_generator.class_indices.keys())

    report = classification_report(true_classes, predicted_classes, target_names=class_labels)
    report_path = os.path.join(model_dir, f'report_{index}.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"Rapport de classification sauvegardé dans : {report_path}")

if __name__ == "__main__":
    data_directory = "/Users/serge/Documents/DATA_ML/PRJ/dataset"
    train_model(data_dir=data_directory, batch_size=16, epochs=50)