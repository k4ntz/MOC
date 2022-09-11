from NicePrinter import *
from pprint import pformat

nicep_availables = {"bbox": bbox, "blue": blue, "bold": bold, "box": box,
                    "cbbox": cbbox, "cbox": cbox, "center": center, "cyan": cyan,
                    "darkcyan": darkcyan, "green": green, "purple": purple,
                    "red": red, "table":table, "title": title,
                    "underline": underline, "yellow": yellow}


def pprint(*args):
    if args[0] in nicep_availables:
        formated_content = ""
        for elem in args[1:]:
            import ipdb; ipdb.set_trace()
            formated_content += pformat(elem)
        print(nicep_availables[args[0]](formated_content))
    else:
        print(*args)
