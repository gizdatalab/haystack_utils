import setuptools

install_requires = [
        "farm-haystack == 1.10",
        "farm-haystack[ocr]==1.10.0",
        "spacy==3.2.0",
        "matplotlib==3.5.1",
        "nltk==3.7",
        "pdfplumber==0.6.2",
        "Pillow==9.1.1",
        "seaborn==0.11.2",
        "transformers==4.21.2",
        "summa==1.2.0",
]


setuptools.setup(
        name='utils',
        version='1.0.1',
        description='Haystack based utils for NLP',
        author='Data Service Center GIZ',
        author_email='prashant.singh@giz.de',
        package_dir={"": "src"},
        packages=setuptools.find_packages(where='src'),  
        install_requires=install_requires, #external packages as dependencies
)
