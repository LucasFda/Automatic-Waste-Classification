import os

def check_train_test_val(base_dir):
    """
    Vérifie la répartition des images dans les dossiers train, test et val pour chaque classe (O, M, P, C).
    Affiche les proportions en pourcentage.

    Args:
        base_dir (str): Chemin vers le répertoire contenant les dossiers train, test, et val.
    """
    # Les classes à vérifier
    classes = ['O', 'M', 'P', 'C']
    splits = ['train', 'test', 'val']

    # Initialisation pour stocker les résultats
    counts = {cls: {split: 0 for split in splits} for cls in classes}
    total_counts = {cls: 0 for cls in classes}

    # Calculer les nombres d'images par classe et par split
    for split in splits:
        split_dir = os.path.join(base_dir, split)
        for cls in classes:
            class_dir = os.path.join(split_dir, cls)
            if os.path.exists(class_dir):
                counts[cls][split] = len(os.listdir(class_dir))
                total_counts[cls] += counts[cls][split]

    # Calculer et afficher les proportions
    for cls in classes:
        print(f"Pour {cls} ({'Organics' if cls == 'O' else 'Metals' if cls == 'M' else 'Plastics' if cls == 'P' else 'Clothes'}):")
        for split in splits:
            if total_counts[cls] > 0:
                percentage = (counts[cls][split] / total_counts[cls]) * 100
            else:
                percentage = 0
            print(f"  {split} : {counts[cls][split]} images ({percentage:.2f}%)")
        print()

if __name__ == "__main__":
    # Chemin vers le dossier 'dataset'
    base_directory = "/Users/serge/Documents/DATA_ML/PRJ/dataset"
    check_train_test_val(base_directory)
