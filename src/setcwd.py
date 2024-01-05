"""Script that sets up the file directory tree before running a notebook.

It uses root of github directory to make sure everyone's code runs from
the same directory, called current working directory cwd.

It adds the python code in the parent directory of the working directory
in the list of paths.

Usage:

import setcwd
setcwd.main()

"""

import os
import subprocess
import sys
import warnings

os.environ["GEOMSTATS_BACKEND"] = "pytorch"


def main():
    """Set up the file paths and directory tree."""
    warnings.filterwarnings("ignore")

    gitroot_path = subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"], universal_newlines=True
    )

    os.chdir(
        os.path.join(gitroot_path[:-1], "src")
    )  # TODO: change "src" if this file is moved.
    print("Working directory: ", os.getcwd())

    sys_dir = os.path.dirname(os.getcwd())
    sys.path.append(sys_dir)
    print("Directory added to path: ", sys_dir)
    sys.path.append(os.getcwd())
    print("Directory added to path: ", os.getcwd())
    h2_dir = os.path.join(sys_dir, "H2_SurfaceMatch")
    sys.path.append(h2_dir)
    print("Directory added to path: ", h2_dir)
    notebook_dir = os.path.join(os.getcwd(), "notebooks")
    print("Directory added to path: ", notebook_dir)
    csv_dir = os.path.join(notebook_dir, "csv")
    print("Directory added to path: ", csv_dir)
