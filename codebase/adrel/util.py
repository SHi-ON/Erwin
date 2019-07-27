"""
Set of general utility functions in Python programming.
"""
import os
import time
import pickle

OUTPUT_DIR = 'outputs'


def timing(t):
    """
    Prints the elapsed time since
    the passed time t to the standard output.

    :param t: the initial time
    """
    elapsed_time = round((time.perf_counter() - t), ndigits=2)
    print('*** Elapsed time:', elapsed_time, 'second(s)')
    return elapsed_time


def pickle_store(store_object, file_name):
    """
    Stores the pickled object to a binary file.

    :param store_object: object to be stored
    :param file_name: the output file name
    """
    file_path = os.path.join(OUTPUT_DIR, file_name + os.extsep + 'pkl')
    with open(file_path, 'wb') as out_handle:
        pickle.dump(store_object, out_handle, pickle.HIGHEST_PROTOCOL)


def pickle_load(file_name):
    """
    Loads a pickle binary file.

    :param file_name: the input file name
    :return: the Python object
    """
    file_path = os.path.join(OUTPUT_DIR, file_name + os.extsep + 'pkl')
    with open(file_path, 'rb') as in_handle:
        return pickle.load(in_handle)
