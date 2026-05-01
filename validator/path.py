from pathlib import Path

def check_path_exists(value: Path) -> Path:
    if not value.exists():
        raise ValueError(f"Path {value} does not exist")
    return value