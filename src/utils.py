import argparse
import os
import pandas as pd
from pandas.io.parsers import ParserError
import tqdm
import ntpath

FALSY_STRINGS = {'off', 'false', '0'}
TRUTHY_STRINGS = {'on', 'true', '1'}

def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("Invalid value for a boolean flag!")
    
def dir_path(path):
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"readable_dir:{path} is not a valid path")
    
def file_path(path):
    if os.path.isfile(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"readable_file:{path} is not a valid path")

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
        
def str2dic(s):
    """`a=x,b=y` to {a:x, b:y}"""
    if s is None :
        return s
    if "," in s:
        params = {}
        for x in s.split(","):
            split = x.split('=')
            assert len(split) == 2
            params[split[0]] = split[1]
    else:
        params = {}
    return AttrDict(params)

def path_leaf(path : str):
    """
    Returns the name of a file given its path
    https://stackoverflow.com/questions/8384737/extract-file-name-from-path-no-matter-what-the-os-path-format
    """
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def get_path(file_item : str, extension : str, save_to : str = None):
    file_name = path_leaf(path = file_item)
    file_path = file_item.split(file_name)[0]
    file_name, ext = os.path.splitext(file_name) 
    if ext.replace(".", "") == extension :
        file_name = file_name + "_" + ext.replace(".", "")
    dest_file = "%s.%s"%(file_name, extension) 
    if os.path.isfile(dest_file):
        i = 1
        while os.path.isfile("%s.%s.%s"%(file_name,str(i),extension)):
            i += 1
        dest_file = "%s.%s.%s"%(file_name,str(i),extension)
    if save_to is not None :
        return os.path.join(save_to, dest_file)
    else :
        return os.path.join(file_path, dest_file)
    
def csv2txt(file_list, text_column, txt_file : None):
    if txt_file is None :
        txt_file = [get_path(f, "txt") for f in file_list]
        result = txt_file
    else :
        if type(txt_file) == str :
            result = [txt_file]
            txt_file = result * len(file_list)
        elif type(txt_file) == list :
            assert len(txt_file) == len(file_list)
            result = [txt_file]
    
    for file_item, tfile in zip(file_list, txt_file):
        with open(tfile, 'a') as f:
            try :
                df = pd.read_csv(file_item)
            except ParserError : # https://stackoverflow.com/questions/33998740/error-in-reading-a-csv-file-in-pandascparsererror-error-tokenizing-data-c-err
                df = pd.read_csv(file_item, lineterminator='\n')
            for row in tqdm.tqdm(list(df.iterrows()), desc="%s" % file_item):
                text = row[1][text_column].strip()
                f.write("%s\n" % text)
                
    return result