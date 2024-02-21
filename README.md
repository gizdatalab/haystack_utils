# Climate Policy Analysis Machine - utils
This is a repo made primarily for NLP tasks and is based mainly on [Haystack](https://docs.haystack.deepset.ai/) and [Hugging face](https://huggingface.co/) already built components.

The tasks performed include:
1. Document processing: Processing the text from docx/text/pdf files and creating the paragraphs list.
2. Search: Performing lexical or semantic search on the paragraphs list created in step 1.
3. SDG Classification: Performing the SDG classification on the paragraphs text.
4. Extracting the keywords based on [Textrank](https://github.com/summanlp/textrank)/TFIDF/[KeyBert](https://github.com/MaartenGr/KeyBERT)

Please use the [colab notebook](https://colab.research.google.com/drive/1ym6Ub5-sMGZkfAF4lnHMWF4MgMpabZ-r?usp=sharing) to get familiar with basic usage of utils
(use branch =main for non-streamlit usage).
For more detailed walkthrough use the [advanced colab notebook](https://colab.research.google.com/drive/1t9ZpcliqlNwkS4NDeKA4JRGdtBKCE9hC?usp=sharing).
There are two branch in the repo. One for using in streamlit environment and another for generic usage like in colab or local machine. 
You can clone the repo for your own use, or also install it as package. 

To install as package (non-streamlit use):
```
pip install -e "git+https://github.com/gizdatalab/haystack_utils.git@main#egg=utils"
```

To install as package for streamlit app:
```
pip install -e "git+https://github.com/gizdatalab/haystack_utils.git@streamlit#egg=utils"
```
To install as package (for CPU-trac Streamlit app https://huggingface.co/spaces/GIZ/cpu_tracs):
```
pip install -e "git+https://github.com/gizdatalab/haystack_utils.git@cputrac#egg=utils"
```
