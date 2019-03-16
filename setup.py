import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="GowersMethod",
    version="1.0.1",
    author="Iain J. Cruickshank",
    author_email="icruicks@andrew.cmu.edu",
    description="Gower's Method for creating Latent Graphs from Multi-Modal Data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ijcruic/Gowers-Method",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'networkx>=2.2',
        'numpy>=1.15.4',
        'python-louvain>=0.11',
        'pandas>=0.23.4'
    ]
)
