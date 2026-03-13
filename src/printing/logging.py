from pathlib import Path

def log(content, path: str | Path):
    """
    Appends the provided content to the end of the file referenced by path.
    """
    
    if isinstance(path, str):
        path = Path(path)

    if path.exists() and path.is_dir():
        raise ValueError("Path points to directory; should be file.")
    
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as file:
        if path.stat().st_size > 0:
            file.write("\n" + content)
        else:
            file.write(content)
