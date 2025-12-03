import os

def get_mats(folder_path: str) -> list[str]:
    """
    Returns a list of the paths to the `.mat` files contained in specified folder.
    """
    if not os.path.isdir(folder_path):
        raise ValueError("The path provided is not a folder path.")

    files = []
    for file in os.listdir(folder_path):
        if file.endswith(".mat"):
            files.append(os.path.join(folder_path, file))
    return files
