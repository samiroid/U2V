import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="U2V", 
    version="0.0.1",
    author="Silvio Amir",
    author_email="silvio.aam@gmail.com",
    description="U2V",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/samiroid/U2V",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
    install_requires=["torch","numpy","allennlp","fasttext","transformers","tqdm"]
)