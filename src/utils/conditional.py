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



@st.cache_resource
def load_conditionalClassifier(config_file:str = None, classifier_name:str = None):
    """
    loads the document classifier using haystack, where the name/path of model
    in HF-hub as string is used to fetch the model object.Either configfile or 
    model should be passed.

    Params
    --------
    config_file: config file path from which to read the model name
    classifier_name: if modelname is passed, it takes a priority if not \
    found then will look for configfile, else raise error.

    Return: Text Classiifcation object with model
    """
    if not classifier_name:
        if not config_file:
            logging.warning("Pass either model name or config file")
            return
        else:
            config = getconfig(config_file)
            classifier_name = config.get('conditional','MODEL')

    logging.info("Loading adapmit classifier")  
    doc_classifier = pipeline("text-classification", 
                            model=classifier_name, top_k =None,
                            token = auth_token,
                            )

    return doc_classifier


@st.cache_data
def adapmit_classification(haystack_doc:pd.DataFrame,
                        threshold:float = 0.5, 
                        classifier_model:pipeline= None
                        )->Tuple[DataFrame,Series]:
    """
    Text-Classification on the list of texts provided. Classifier provides the 
    most appropriate Conditionality label for each text.

    Params
    ---------
    haystack_doc: The output of Tapp_extraction
    threshold: threshold value for the model to keep the results from classifier
    classifiermodel: you can pass the classifier model directly,which takes priority
    however if not then looks for model in streamlit session.
    In case of streamlit avoid passing the model directly.

    Returns
    ----------
    df: Dataframe, with columns added ['ConditionalLabel','UnconditionalLabel']
    """
    logging.info("Working on Conditionality")
    if not classifier_model:
        classifier_model = st.session_state['conditional_classifier']    

    # predict classes
    results = classifier_model(list(haystack_doc.text))
    # extract score for each class and create dataframe
    labels_= [{label['label']:round(label['score'],3) for label in result} 
                                                    for result in results]
    df1 = pd.DataFrame(labels_)
    label_names = list(df1.columns)
    # conver the dataframe into truth value dataframe rather than probabilities
    df2 = df1 >= threshold
    # append the dataframe to original dataframe 
    df = pd.concat([haystack_doc,df2], axis=1)

    return df