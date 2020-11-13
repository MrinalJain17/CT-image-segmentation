from setuptools import find_packages, setup

setup(
    name="capstone_project",
    version="0.0.0",
    description="Capstone Project",
    author="Mrinal Jain",
    author_email="mrinal.jain@nyu.edu",
    url="https://github.com/MrinalJain17/CT-image-segmentation",
    install_requires=[],
    packages=find_packages(exclude=["storage"]),
)
