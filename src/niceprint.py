from NicePrinter import *
from pprint import pformat
from argparse import Namespace
import os


def box(string):
    nb_line = string.count("\n")
    string = string.replace('\t', '')
    max_length = max([len(line) for line in string.split("\n")])
    boxed = "-" * (max_length + 2) + "\n"
    for line in string.split("\n"):
        boxed += '|' + line + ' '*(max_length-len(line)) +  '|\n'
    return boxed + "-" * (max_length + 2)


nicep_availables = {"bbox": bbox, "blue": blue, "bold": bold, "box": box,
                    "cbbox": cbbox, "cbox": cbox, "center": center, "cyan": cyan,
                    "darkcyan": darkcyan, "green": green, "purple": purple,
                    "red": red, "table":table, "title": title,
                    "underline": underline, "yellow": yellow}


def pprint(*args):
    try:
        if args[0] in nicep_availables:
            formating_funcs = []
            while args[0] in nicep_availables:
                formating_funcs.append(nicep_availables[args[0]])
                args = args[1:]
            formated_content = ""
            for elem in args:
                if isinstance(elem, Namespace):
                    formated_content += "\n"
                    for key, attr in elem._get_kwargs():
                        formated_content += f"\t{key}: {attr}\n"
                elif isinstance(elem, str):
                    formated_content += elem
                else:
                    formated_content += pformat(elem)
            for formfunc in reversed(formating_funcs):
                formated_content = formfunc(formated_content)
            print(formated_content)
        else:
            print(*args)
    except TypeError:
        print(args)


def makedirs(*args, **kwargs):
    path_to_create = os.path.join(*args)
    if not os.path.exists(path_to_create):
        pprint("purple", "center", "box", f"Created path {path_to_create}")
    os.makedirs(path_to_create, exist_ok=True)
