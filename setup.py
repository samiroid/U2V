import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="user2vec", 
    version="0.0.1",
    author="Silvio Amir",
    author_email="silvio.aam@gmail.com",
    description="User2Vec",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/samiroid/user2vec_torch",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
    install_requires=["torch","numpy"]
)