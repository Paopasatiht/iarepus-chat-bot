import numpy as np
import string


def _float_converter(str_vec : str):
    """ Convert a "string" in pandas to "float" 
    ref : https://stackoverflow.com/questions/65124688/convert-string-to-numpy-array-python
    
    Parameters : 
        str_vec : string
                : A list of list of float that was convert in the term of string in csv files
        float_vector : list
                : A list of list of float (Ex : [[1.540, 1.374, 7.129 ... , ... , 4.567]]) dim = 756
    """
    _float_vector = np.fromstring(str_vec.replace('[', '').replace(']', '').replace('\n', ''), dtype=float, sep=' ')

    return [_float_vector] 