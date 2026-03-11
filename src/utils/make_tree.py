import os

OUTPUT_FILE = "project_structure.txt"

IGNORE_DIRS = {
    ".venv",
    ".git",
    "__pycache__",
    ".idea"
}

def make_tree(start_path=".", prefix=""):
    entries = sorted(os.listdir(start_path))
    entries = [e for e in entries if e not in IGNORE_DIRS]

    for i, entry in enumerate(entries):
        path = os.path.join(start_path, entry)
        connector = "|___ "

        line = prefix + connector + entry
        f.write(line + "\n")

        if os.path.isdir(path):
            make_tree(path, prefix + "|    ")

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    root_name = os.path.basename(os.getcwd())
    f.write(root_name + "\n")
    make_tree("../..")

print(f"Project structure saved to {OUTPUT_FILE}")