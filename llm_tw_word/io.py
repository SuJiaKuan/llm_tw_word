import json
import pathlib


def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def save_json(file_path, data, indent=2):
    with open(file_path, "w", encoding="utf8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def mkdir_p(dir_path):
    pathlib.Path(dir_path).mkdir(parents=True, exist_ok=True)
