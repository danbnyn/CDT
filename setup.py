# delaunay_triangulation/setup.py

from setuptools import setup, find_packages

# Helper function to read dependencies from requirements.txt
def parse_requirements(filename):
    with open(filename, 'r') as file:
        return [line.strip() for line in file if line and not line.startswith("#")]

setup(
    name="delaunay_triangulation",
    version="0.1",
    description="A Python package for Constrained Delaunay Triangulation on arbitrary 2D surfaces",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Dan Benayoun",
    author_email="dnbenayoun@gmail.com",
    url="https://github.com/danbnyn/CDT",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=parse_requirements("requirements.txt"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
