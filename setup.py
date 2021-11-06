import os
import sys
from os import path
from shutil import rmtree

from setuptools import Command, find_packages, setup

here = os.path.abspath(os.path.dirname(__file__))


class UploadCommand(Command):
    """Support setup.py upload.
    Adapted from https://github.com/robustness-gym/meerkat.
    """

    description = "Build and publish the package."
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print("\033[1m{0}\033[0m".format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status("Removing previous builds…")
            rmtree(os.path.join(here, "dist"))
        except OSError:
            pass

        self.status("Building Source and Wheel (universal) distribution…")
        os.system("{0} setup.py sdist bdist_wheel --universal".format(sys.executable))

        self.status("Uploading the package to PyPI via Twine…")
        os.system("twine upload dist/*")

        self.status("Pushing git tags…")
        os.system("git tag v{0}".format(get_version()))
        os.system("git push --tags")

        sys.exit()


def get_version():
    init_py_path = path.join(path.abspath(path.dirname(__file__)), "skm_tea", "__init__.py")
    init_py = open(init_py_path, "r").readlines()
    version_line = [l.strip() for l in init_py if l.startswith("__version__")][0]  # noqa: E741
    version = version_line.split("=")[-1].strip().strip("'\"")

    return version


setup(
    name="skm-tea",
    version=get_version(),
    author="The SKM-TEA team",
    url="",
    description="A package for training on, interacting with, and visualizing the SKM-TEA dataset",
    packages=find_packages(exclude=("configs", "tests", "*.tests", "*.tests.*", "tests.*")),
    python_requires=">=3.6",
    install_requires=[
        "h5py",
        "matplotlib",
        "numpy",
        "fvcore",
        "dosma>=0.1.0",
        "iopath",
        "medpy",
        "ismrmrd",
        "pandas",
        "tqdm",
        "xlrd",
        "monai>=0.3.0",
        "meddlr",
        "pytorch-lightning>=1.0.0",
        "zarr",
    ],
    extras_require={
        "dev": [
            "flake8",
            "isort",
            "black",
            "flake8-bugbear",
            "flake8-comprehensions",
            "pre-commit",
        ]
    },
)
