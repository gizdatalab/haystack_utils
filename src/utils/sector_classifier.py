from typing import List, Tuple
from typing_extensions import Literal
import logging
import pandas as pd
from pandas import DataFrame, Series
from utils.config import getconfig
import streamlit as st
from transformers import pipeline


@st.cache_resource
def load_sectorClassifier(config_file:str = None, classifier_name:str = None):
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
    Return: document classifier model
    """
    if not classifier_name:
        if not config_file:
            logging.warning("Pass either model name or config file")
            return
        else:
            config = getconfig(config_file)
            classifier_name = config.get('sector','MODEL')
    
    logging.info("Loading sector classifier")
    # we are using the pipeline as the model is multilabel and DocumentClassifier 
    # from Haystack doesnt support multilabel
    # in pipeline we use 'sigmoid' to explicitly tell pipeline to make it multilabel
    # if not then it will automatically use softmax, which is not a desired thing.
    # doc_classifier = TransformersDocumentClassifier(
    #                     model_name_or_path=classifier_name,
    #                     task="text-classification",
    #                     top_k = None)

    doc_classifier = pipeline("text-classification", 
                            model=classifier_name, 
                            return_all_scores=True, 
                            function_to_apply= "sigmoid")

    return doc_classifier


@st.cache_data
def sector_classification(haystack_doc:pd.DataFrame,
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
    df: Dataframe
    """
    logging.info("Working on Sector Identification")
    haystack_doc['Sector Label'] = 'NA'
    if not classifier_model:
        classifier_model = st.session_state['sector_classifier']
    
        predictions = classifier_model(list(haystack_doc.text))

    # getting the sector label and scores
    list_ = []
    for i in range(len(predictions)):

      temp = predictions[i]
      placeholder = {}
      for j in range(len(temp)):
        placeholder[temp[j]['label']] = temp[j]['score']
      list_.append(placeholder)
    labels_ = [{**list_[l]} for l in range(len(predictions))]
    truth_df = DataFrame.from_dict(labels_)
    truth_df = truth_df.round(2)
    # based on threshold value, we convert each sector score into boolean
    truth_df = truth_df.astype(float) >= threshold
    truth_df = truth_df.astype(str)
    # collecting list of Sector Labels
    categories = list(truth_df.columns)
    # we collect the Sector Labels as set, None represent the value at the index
    # in the list of Sector Labels.
    truth_df['Sector Label'] = truth_df.apply(lambda x: {i if x[i]=='True' else 
                                              None for i in categories}, axis=1)
    # we keep all Sector label except None                                          
    truth_df['Sector Label'] = truth_df.apply(lambda x: list(x['Sector Label'] 
                                                            -{None}),axis=1)
    haystack_doc['Sector Label'] = list(truth_df['Sector Label'])
    return haystack_doc