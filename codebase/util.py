"""
Set of general utility functions in Python development.
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
    t2 = time.perf_counter()
    elapsed_time = round((t2 - t), ndigits=3)
    print('*** Elapsed time: {0:.3f} second(s)'.format(elapsed_time))
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


def erwin_log(report):
    print('ErwinLog: An exception has been caught:\n', report)
