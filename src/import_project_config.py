"""Gives functions in src the ability to access the default_config of the project folder that called the src function."""

import importlib.util
import inspect
import os


def import_default_config(project_dir):
    """Recovers default_config in the directory of the calling script's path."""
    default_config_path = os.path.join(project_dir, "default_config.py")

    # Check if the file exists before attempting to import
    if os.path.isfile(default_config_path):
        spec = importlib.util.spec_from_file_location(
            "default_config", default_config_path
        )
        default_config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(default_config_module)
        return default_config_module
    else:
        raise ImportError(
            "default_config module not found in the calling script's directory"
        )
