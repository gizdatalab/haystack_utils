from typing import List, Tuple
from typing_extensions import Literal
import logging
import pandas as pd
from pandas import DataFrame, Series
from utils.config import getconfig
import streamlit as st
from setfit import SetFitModel
import os
auth_token = os.environ.get("privatemodels") or True

## Labels dictionary ###
sectors = ['GHGLabel','NetzeroLabel','NonGHGLabel' ]


@st.cache_resource
def load_subtargetClassifier(config_file:str = None, classifier_name:str = None):
    """
    loads the document classifier using haystack, where the name/path of model
    in HF-hub as string is used to fetch the model object.Either configfile or 
    model should be passed.

    Params
    --------
    config_file: config file path from which to read the model name
    classifier_name: if modelname is passed, it takes a priority if not \
    found then will look for configfile, else raise error.

    Return: document setfit model
    """
    if not classifier_name:
        if not config_file:
            logging.warning("Pass either model name or config file")
            return
        else:
            config = getconfig(config_file)
            classifier_name = config.get('subtarget','MODEL')

    logging.info("Loading setfit subtarget classifier")   
    doc_classifier = SetFitModel.from_pretrained(classifier_name)
    return doc_classifier


@st.cache_data
def subtarget_classification(haystack_doc:pd.DataFrame,
                        threshold:float = 0.5, 
                        classifier_model:SetFitModel= None
                        )->Tuple[DataFrame,Series]:
    """
    Text-Classification on the list of texts provided. Classifier provides the 
    most appropriate Subtarget label for each text. limited to as defined in subtarget list

    Params
    ---------
    haystack_doc: The output of Tapp_extraction
    threshold: threshold value for the model to keep the results from classifier
    classifiermodel: you can pass the classifier model directly,which takes priority
    however if not then looks for model in streamlit session.
    In case of streamlit avoid passing the model directly.

    Returns
    ----------
    df: Dataframe, with columns added ['GHGLabel','NetzeroLabel','NonGHGLabel']
    """
    logging.info("Working on Subtarget Identification")
    if not classifier_model:
        classifier_model = st.session_state['subtarget_classifier']    
    
    predictions = classifier_model(list(haystack_doc.text))

    # getting the sector label Boolean flag, we will not use threshold value, 
    # but use default of 0.5
    list_ = []
    for i in range(len(predictions)):
      temp = predictions[i]
      placeholder = {}
      for idx,sector in enumerate(sectors):
        placeholder[sector] = bool(temp[idx])
      list_.append(placeholder)
    truth_df = pd.DataFrame(list_)

    # we collect the Sector Labels as set, None represent the value at the index
    df = pd.concat([haystack_doc,truth_df],axis=1)
    return df