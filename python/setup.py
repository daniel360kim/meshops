from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

VERSION = '0.1'
DESCRIPTION = 'Library for dealing with meshes and stuff'
LONG_DESCRIPTION = \
    'idk yet (wip)'

# Setting up
setup(
    name="meshthingy",
    version=VERSION,
    author="nonagon",
    author_email="donaldcenaaa@outlook.com",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=find_packages(),
    install_requires=['numpy', "pillow", "torch"], # Add dependencies here
    keywords=['python', 'tensor', 'mesh', 'meshgrid', 'simulation', 'tensor operations'],
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)