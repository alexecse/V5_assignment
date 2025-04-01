import os
import shutil

def delete_if_exists(path):
    if os.path.exists(path) and os.path.isdir(path):
        print(f"Deleting: {path}")
        shutil.rmtree(path)
    else:
        print(f"Not found or not a directory: {path}")

def clean_output_and_statistics(base_path="."):
    # Define the target folders
    output_dir = os.path.join(base_path, "output")
    statistics_dir = os.path.join(base_path, "statistics")

    # Delete them if they exist
    delete_if_exists(output_dir)
    delete_if_exists(statistics_dir)

if __name__ == "__main__":
    clean_output_and_statistics()
