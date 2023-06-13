"""Script that sets up the file directory tree before running a notebook.

Usage:

import setcwd
setcwd.main()

"""

import os
import subprocess
import sys
import warnings


def main():
    warnings.filterwarnings("ignore")

    gitroot_path = subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"], universal_newlines=True
    )

    os.chdir(os.path.join(gitroot_path[:-1], "my28brains"))
    print("Working directory: ", os.getcwd())

    sys_dir = os.path.dirname(os.getcwd())
    sys.path.append(sys_dir)
    print("Directory added to path: ", sys_dir)
    sys.path.append(os.getcwd())
    print("Directory added to path: ", os.getcwd())
