import configparser
import logging

def getconfig(configfile_path:str):
    """
    configfile_path: file path of .cfg file
    """

    config = configparser.ConfigParser()

    try:
        config.read_file(open(configfile_path))
        return config
    except:
        logging.warning("config file not found")


# read all the necessary variables
def get_classifier_params(model_name):
    config = getconfig('paramconfig.cfg')
    params = {}
    params['model_name'] = config.get(model_name,'MODEL')
    params['split_by'] = config.get(model_name,'SPLIT_BY')
    params['split_length'] = int(config.get(model_name,'SPLIT_LENGTH'))
    params['split_overlap'] = int(config.get(model_name,'SPLIT_OVERLAP'))
    params['remove_punc'] = bool(int(config.get(model_name,'REMOVE_PUNC')))
    params['split_respect_sentence_boundary'] = bool(int(config.get(model_name,'RESPECT_SENTENCE_BOUNDARY')))
    params['threshold'] = float(config.get(model_name,'THRESHOLD'))
    params['top_n'] = int(config.get(model_name,'TOP_KEY'))

    return params