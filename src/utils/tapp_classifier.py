from typing import List, Tuple
from typing_extensions import Literal
import logging
import pandas as pd
from pandas import DataFrame, Series
from utils.config import getconfig
import streamlit as st
from transformers import pipeline
import os
auth_token = os.environ.get("privatemodels") or True

## Labels dictionary ###
_lab_dict = {
            'NEGATIVE':'NO TARGET INFO',
            'TARGET':'TARGET',
            }

@st.cache_resource
def load_targetClassifier(config_file:str = None, classifier_name:str = None):
    """
    loads the document classifier using haystack, where the name/path of model
    in HF-hub as string is used to fetch the model object.Either configfile or 
    model should be passed.
    1. https://docs.haystack.deepset.ai/reference/document-classifier-api
    2. https://docs.haystack.deepset.ai/docs/document_classifier
    Params
    --------
    config_file: config file path from which to read the model name
    classifier_name: if modelname is passed, it takes a priority if not \
    found then will look for configfile, else raise error.
    Return: Transformer text Classification pipeline object
    """
    if not classifier_name:
        if not config_file:
            logging.warning("Pass either model name or config file")
            return
        else:
            config = getconfig(config_file)
            classifier_name = config.get('target','MODEL')
    
    logging.info("Loading classifier")  
      
    doc_classifier = pipeline("text-classification", 
                            model=classifier_name, top_k =None,
                            token = auth_token,
                            )

    return doc_classifier


@st.cache_data
def target_classification(haystack_doc:pd.DataFrame,
                        threshold:float = 0.5, 
                        classifier_model:pipeline= None
                        )->Tuple[DataFrame,Series]:
    """
    Text-Classification on the list of texts provided. Classifier provides the 
    most appropriate label for each text. these labels are in terms of if text 
    belongs to which particular Sustainable Devleopment Goal (SDG).
    Params
    ---------
    haystack_doc: List of haystack Documents. The output of Preprocessing Pipeline 
    contains the list of paragraphs in different format,here the list of 
    Haystack Documents is used.
    threshold: threshold value for the model to keep the results from classifier
    classifiermodel: you can pass the classifier model directly,which takes priority
    however if not then looks for model in streamlit session.
    In case of streamlit avoid passing the model directly.
    Returns
    ----------
    df: Dataframe with two columns['SDG:int', 'text']
    x: Series object with the unique SDG covered in the document uploaded and 
    the number of times it is covered/discussed/count_of_paragraphs. 
    """
    logging.info("Working on Target Extraction")
    if not classifier_model:
        classifier_model = st.session_state['target_classifier']
    
    # predict classes
    results = classifier_model(list(haystack_doc.text))
    # extract score for each class and create dataframe
    labels_= [{label['label']:round(label['score'],3) for label in result} 
                                                    for result in results]
    df1 = pd.DataFrame(labels_)
    # conver the dataframe into truth value dataframe rather than probabilities
    df2 = df1 >= threshold
    # append the dataframe to original dataframe
    df = pd.concat([haystack_doc,df2],axis=1)
    # making index to start from 1 rather than 0
    df.index += 1
    return df