from typing import List, Tuple
from typing_extensions import Literal
import logging
import pandas as pd
from pandas import DataFrame, Series
from utils.config import getconfig
import streamlit as st
from transformers import pipeline
from setfit import SetFitModel

## Labels dictionary ###
sectors = [
            'Economy-wide',
            'Energy',
            'Other Sector',
            'Transport',
              ]


@st.cache_resource
def load_sectorClassifier(config_file:str = None, classifier_name:str = None):
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
            classifier_name = config.get('sector','MODEL')

    logging.info("Loading setfit sector classifier")   
    doc_classifier = SetFitModel.from_pretrained(classifier_name)
    return doc_classifier


@st.cache_data
def sector_classification(haystack_doc:pd.DataFrame,
                        threshold:float = 0.5, 
                        classifier_model:SetFitModel= None
                        )->Tuple[DataFrame,Series]:
    """
    Text-Classification on the list of texts provided. Classifier provides the 
    most appropriate Sector label for each text. limited to as defined in _sector_dict

    Params
    ---------
    haystack_doc: The output of Tapp_extraction
    threshold: threshold value for the model to keep the results from classifier
    classifiermodel: you can pass the classifier model directly,which takes priority
    however if not then looks for model in streamlit session.
    In case of streamlit avoid passing the model directly.

    Returns
    ----------
    df: Dataframe, with columns added ['Economy-wide','Energy','Other Sector','Transport']
    """
    logging.info("Working on Sector Identification")
    if not classifier_model:
        classifier_model = st.session_state['sector_classifier']    
        predictions = classifier_model(list(haystack_doc.text))
    st.write(predictions)

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