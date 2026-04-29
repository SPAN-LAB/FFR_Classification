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
            file.write("\n" + f"{content}")
        else:
            file.write(f"{content}")

def is_empty(path: str | Path):
    """
    Checks if the file referenced by path does not have any content, 
    returning true when the above statement is true.
    """
    
    if isinstance(path, str):
        path = Path(path)
    
    if not path.exists():
        raise ValueError("Path does not exist.")
    
    if path.is_dir():
        raise ValueError("Path points to directory; should be file.")
    
    if path.stat().st_size > 0:
        return False
    return True
