import os

def rename_images_by_class(base_dir):
    """
    Renomme les images dans chaque dossier de classes (O, M, P, C) avec un format : nom_du_dossier_numéro.
    
    Args:
        base_dir (str): Chemin vers le répertoire contenant les dossiers de classes.
    """
    # Liste des dossiers/classes
    classes = ['O', 'M', 'P', 'C']

    for cls in classes:
        class_dir = os.path.join(base_dir, cls)
        if not os.path.exists(class_dir):
            print(f"Le dossier {class_dir} n'existe pas.")
            continue
        
        print(f"Renommage des images dans le dossier {class_dir}...")

        # Parcourt les images et les renomme
        for idx, filename in enumerate(os.listdir(class_dir), start=1):
            file_path = os.path.join(class_dir, filename)
            if os.path.isfile(file_path):  # Vérifie que c'est bien un fichier
                # Extension du fichier (jpg, png, etc.)
                extension = os.path.splitext(filename)[1]
                # Nouveau nom de fichier
                new_name = f"{cls}_{idx}{extension}"
                new_path = os.path.join(class_dir, new_name)
                # Renommage
                os.rename(file_path, new_path)
        print(f"Renommage terminé pour le dossier {cls}.")

if __name__ == "__main__":
    # Chemin vers le dossier 'dataset'
    base_directory = "dataset"
    rename_images_by_class(base_directory)
