#!/usr/bin/env python3
"""
Simple script to clear Python cache files (.pyc, .pyo, and __pycache__ directories)
"""

import os
import shutil
import sys


def clear_python_cache(directory="."):
    """Clear Python cache files and directories."""
    cache_items_removed = 0

    for root, dirs, files in os.walk(directory):
        # Remove __pycache__ directories
        if "__pycache__" in dirs:
            pycache_path = os.path.join(root, "__pycache__")
            try:
                shutil.rmtree(pycache_path)
                print(f"Removed directory: {pycache_path}")
                cache_items_removed += 1
            except Exception as e:
                print(f"Failed to remove {pycache_path}: {e}")

        # Remove .pyc and .pyo files
        for file in files:
            if file.endswith((".pyc", ".pyo")):
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    print(f"Removed file: {file_path}")
                    cache_items_removed += 1
                except Exception as e:
                    print(f"Failed to remove {file_path}: {e}")

    print(f"\nCache cleanup complete! Removed {cache_items_removed} items.")


if __name__ == "__main__":
    directory = sys.argv[1] if len(sys.argv) > 1 else "."
    print(f"Clearing Python cache in: {os.path.abspath(directory)}")
    clear_python_cache(directory)
