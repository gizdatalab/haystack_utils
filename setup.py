import setuptools

install_requires = [
        "farm-haystack == 1.10",
        "farm-haystack[ocr]==1.10.0",
        "spacy==3.2.0",
        "matplotlib==3.5.1",
        "nltk==3.7",
        "numpy==1.21.6",
        "pandas==1.2.0",
        "pdfplumber==0.6.2",
        "Pillow==9.1.1",
        "seaborn==0.11.2",
        "transformers==4.21.2",
        "st-annotated-text==3.0.0",
        "markdown==3.4.1",
        "summa==1.2.0",
]


setuptools.setup(
   name='utils',
   version='1.0',
   description='A useful module',
   author='prashant',
   author_email='prashant.singh@giz.de',
   packages=setuptools.find_packages(where='utils'),  #same as name
   install_requires=install_requires, #external packages as dependencies
)