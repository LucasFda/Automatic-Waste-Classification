import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tkinter import Tk, filedialog

def preprocess_image(image_path, target_size=(150, 150)):
    """
    Charge et prétraite une image pour le modèle.

    Args:
        image_path (str): Chemin de l'image.
        target_size (tuple): Dimensions cibles de l'image.

    Returns:
        np.array: Image prétraitée pour le modèle.
        np.array: Image originale.
    """
    original_image = cv2.imread(image_path)
    if original_image is None:
        raise ValueError(f"Image non trouvée : {image_path}")

    resized_image = cv2.resize(original_image, target_size)  # Redimensionner l'image
    normalized_image = resized_image / 255.0  # Normalisation
    expanded_image = np.expand_dims(normalized_image, axis=0)  # Ajouter la dimension batch
    return expanded_image, original_image

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
    files = [f for f in os.listdir(directory) if f.startswith('img_result_') and f.endswith('.png')]
    if not files:
        return 1
    indexes = [int(f.split('_')[2].split('.')[0]) for f in files]
    return max(indexes) + 1

def predict_image(model_path, class_labels):
    """
    Prédit la classe d'une image sélectionnée par l'utilisateur.

    Args:
        model_path (str): Chemin vers le modèle entraîné.
        class_labels (list): Liste des classes avec leurs noms.
    """
    # Charger le modèle Keras
    model = load_model(model_path)
    print(f"Modèle chargé : {model_path}")

    # Sélectionner une image via une boîte de dialogue
    Tk().withdraw()
    image_path = filedialog.askopenfilename(
        title="Sélectionnez une image",
        filetypes=[("Images", "*.jpg *.jpeg *.png")]
    )

    if not image_path:
        print("Aucune image sélectionnée.")
        return

    # Préparer l'image pour le modèle
    preprocessed_image, original_image = preprocess_image(image_path)

    # Prédire la classe
    predictions = model.predict(preprocessed_image)[0]
    predicted_index = np.argmax(predictions)
    predicted_label = class_labels[predicted_index]
    predicted_confidence = predictions[predicted_index]

    # Annoter l'image avec la prédiction
    result_text = f"{predicted_label}: {predicted_confidence:.2f}"
    font_scale = max(0.5, min(original_image.shape[1] / 800, 1.5))
    font_thickness = max(1, int(font_scale * 2))
    text_size = cv2.getTextSize(result_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
    text_x, text_y = 10, text_size[1] + 10
    cv2.rectangle(original_image, (text_x - 5, text_y - text_size[1] - 5), 
                  (text_x + text_size[0] + 5, text_y + 5), (0, 0, 0), -1)
    cv2.putText(original_image, result_text, (text_x, text_y), 
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), font_thickness)

    # Afficher l'image avec la prédiction
    cv2.imshow("Résultat", original_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Sauvegarder l'image annotée dans le dossier results
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    index = get_next_index(results_dir)
    result_path = os.path.join(results_dir, f"img_result_{index}.png")
    cv2.imwrite(result_path, original_image)
    print(f"Image annotée sauvegardée dans : {result_path}")

if __name__ == "__main__":
    # Chemin vers le modèle (mettre le chemin correct)
    model_path = "./models/model_8.keras"

    # Vérifiez l'ordre des classes pour correspondre à votre modèle
    class_labels = ["Clothes", "Metals", "Organics", "Plastics"]  # Ajustez l'ordre si nécessaire

    # Lancer la prédiction
    predict_image(model_path, class_labels)





