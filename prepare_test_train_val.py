import os
import random
import shutil

def prepare_test_train_val(base_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Prépare les dossiers train, test et val à partir d'un dataset existant.
    Les images des classes O, P, M et C sont réparties aléatoirement.

    Args:
        base_dir (str): Chemin vers le répertoire contenant les dossiers de classes (O, P, M, C).
        train_ratio (float): 70% - Proportion des images pour l'ensemble train.
        val_ratio (float): 15% - Proportion des images pour l'ensemble val.
        test_ratio (float): 15% - Proportion des images pour l'ensemble test.
    """
    # Vérifie que les proportions sont correctes
    if train_ratio + val_ratio + test_ratio != 1.0:
        raise ValueError("Les proportions train, val et test doivent totaliser 1.0.")

    # Liste des classes (dossiers)
    classes = ['O', 'P', 'M', 'C']

    # Crée les dossiers train, test et val
    for split in ['train', 'val', 'test']:
        for cls in classes:
            split_dir = os.path.join(base_dir, split, cls)
            os.makedirs(split_dir, exist_ok=True)

    # Répartition des images
    for cls in classes:
        class_dir = os.path.join(base_dir, cls)
        if not os.path.exists(class_dir):
            print(f"Le dossier {class_dir} n'existe pas. Ignoré.")
            continue
        
        images = os.listdir(class_dir)
        random.shuffle(images)

        # Calcul des indices pour chaque split
        train_split = int(train_ratio * len(images))
        val_split = int((train_ratio + val_ratio) * len(images))

        # Répartition des images dans train, val et test
        for idx, img in enumerate(images):
            src_path = os.path.join(class_dir, img)
            if not os.path.isfile(src_path):
                continue  # Ignore les sous-dossiers ou fichiers invalides
            
            if idx < train_split:
                dst_path = os.path.join(base_dir, 'train', cls, img)
            elif idx < val_split:
                dst_path = os.path.join(base_dir, 'val', cls, img)
            else:
                dst_path = os.path.join(base_dir, 'test', cls, img)
            
            shutil.move(src_path, dst_path)
        
        print(f"Répartition terminée pour la classe {cls}.")

    print("Préparation des dossiers train, val et test terminée.")

if __name__ == "__main__":
    # Chemin vers le dossier 'dataset'
    base_directory = "/Users/serge/Documents/DATA_ML/PRJ/dataset"
    prepare_test_train_val(base_directory)
