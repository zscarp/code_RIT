"""
################################################################################
CSCI 633: Biologically-Inspired Intelligent Systems
Version taught by Alexander G. Ororbia II

Miscellaneous utility functions.
################################################################################
"""

import functools

def register_metadata(key, value):
    """
    Provides a way to attach metadata to a function.

    # Arguments
    * `key` - The attribute name for the metadata.
    * `value` - The value of the metadata.

    # Source
    Key components taken from stack overflow post: https://stackoverflow.com/a/14851408
    """
    def wrapped(fn):
        @functools.wraps(fn)
        def wrapped_f(*args, **kwargs):
            return fn(*args, **kwargs)
        if not hasattr(wrapped_f, "metadata"):
            wrapped_f.metadata = {}
        wrapped_f.metadata[key] = value
        return wrapped_f
    return wrapped