import importlib

def load(module):
    wrapper_module = importlib.import_module(
        "tacorn." + module + "_wrapper")
    return wrapper_module
 