from typing import List, Tuple
from typing_extensions import Literal
import logging
import pandas as pd
from pandas import DataFrame, Series
from utils.config import getconfig
import streamlit as st
from transformers import pipeline
from setfit import SetFitModel
import os
auth_token = os.environ.get("privatemodels") or True


@st.cache_resource
def load_tappClassifier(config_file:str = None, classifier_name:str = None):
    """
    loads the document classifier using haystack, where the name/path of model
    in HF-hub as string is used to fetch the model object.Either configfile or 
    pipeline with model object should be passed.

    Params
    --------
    config_file: config file path from which to read the model name
    classifier_name: if modelname is passed, it takes a priority if not \
    found then will look for configfile, else raise error.
    --------
    Return: Transformer text Classification pipeline object
    """
    if not classifier_name:
        if not config_file:
            logging.warning("Pass either model name or config file")
            return
        else:
            config = getconfig(config_file)
            classifier_name = config.get('tapp','MODEL')
    
    logging.info("Loading tapp classifier")  
      
    doc_classifier = pipeline("text-classification", 
                            model=classifier_name, top_k =None,
                            token = auth_token,
                            )

    return doc_classifier

@st.cache_resource
def load_targetClassifier(config_file:str = None, classifier_name:str = None):
    """
    loads the document classifier using haystack, where the name/path of model
    in HF-hub as string is used to fetch the model object.Either configfile or 
    Setfitmodel should be passed.

    Params
    --------
    config_file: config file path from which to read the model name
    classifier_name: if modelname is passed, it takes a priority if not \
    found then will look for configfile, else raise error.
    ------------
    Return: Setfitmodel

    """
    if not classifier_name:
        if not config_file:
            logging.warning("Pass either model name or config file")
            return
        else:
            config = getconfig(config_file)
            classifier_name = config.get('target','MODEL')
    
    logging.info("Loading setfit target classifier")   
    doc_classifier = SetFitModel.from_pretrained(classifier_name)
    return doc_classifier

@st.cache_data
def tapp_classification(haystack_doc:pd.DataFrame,
                        threshold:float = 0.5, 
                        tapp_classifier_model:pipeline= None,
                        target_setfit:SetFitModel = None,
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
    tapp_classifiermodel: you can pass the classifier model directly,which takes priority
    however if not then looks for model in streamlit session. This will classify if text 
    is Target/Action/Policy/Plan
    target_setfit: this classifier will use IKI specific target definition to update the
    Target identification in dataframe. In case of streamlit avoid passing the model directly.


    Returns
    ----------
    df: Dataframe with two columns[text, pagenumber, TargetLabel, ActionLabel, PolicyLabel,
                                    PlanLabel] 
    """
    logging.info("Working on TAPP Extraction")
    if not tapp_classifier_model:
        tapp_classifier_model = st.session_state['tapp_classifier']
    
    # predict classes
    results = tapp_classifier_model(list(haystack_doc.text))
    # extract score for each class and create dataframe
    labels_= [{label['label']:round(label['score'],3) for label in result} 
                                                    for result in results]
    df1 = pd.DataFrame(labels_)
    label_names = list(df1.columns)
    # conver the dataframe into truth value dataframe rather than probabilities
    df2 = df1 >= threshold
    # append the dataframe to original dataframe 
    df = pd.concat([haystack_doc,df2],ignore_index=True, axis=1)
    st.dataframe(df)
    # we drop the Target from Tapp to fetch Target more suited to IKI-tracs taxonomy
    df.drop('TargetLabel',axis=1, inplace=True)
    if not target_setfit:
        target_setfit = st.session_state['target_classifier']
    results = target_setfit(list(df.text))
    results = [True if x=='TARGET' else False for x in results]
    df['TargetLabel'] = results
    df['check'] = df.apply(lambda x: any([x[label] for label in label_names]),axis=1)
    df = df[df.check == True].reset_index(drop=True)
    df.drop('check',axis=1, inplace=True)

    # making index to start from 1 rather than 0
    df.index += 1
    return df