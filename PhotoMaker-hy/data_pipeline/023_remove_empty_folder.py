import os

def delete_empty_folders(folder_path):
    for folder in os.listdir(folder_path):
        folder_path_full = os.path.join(folder_path, folder)
        if os.path.isdir(folder_path_full):
            try:
                os.rmdir(folder_path_full)
                print("Deleted empty folder:", folder_path_full)
            except OSError as e:
                if e.errno == 39:  # Directory not empty
                    print("Not an empty folder:", folder_path_full)
                else:
                    print("Error deleting folder:", folder_path_full)

folder_path = "data/poco_celeb_images_cropped_1024"

delete_empty_folders(folder_path)
