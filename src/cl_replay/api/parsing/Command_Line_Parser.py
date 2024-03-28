import sys


class Command_Line_Parser(object):
    ''' Reads in command line arguments. '''
    
    def __init__(self):
        self.command_line_parameter_list = sys.argv[1:]


    # method made to look like the method in argparse objects.
    # however, it does not really parse the cmd line, just identifies
    # parameter name strings that start with --, and adds all strings
    # until the next -- string as arguments
    def parse_args(self):
        unknown = self.command_line_parameter_list
        unknownDict = {}
        pName = ""
        for item in unknown:
            if '--' in item:
                # check of valuelist if previous param is one-elem
                if pName != "":
                    lst = unknownDict[pName]
                    if len(lst) == 1:
                        unknownDict[pName] = lst[0]
                pName = item[2:]
                unknownDict[pName] = []
            else:
                unknownDict[pName].append(item)

        if pName != "":
            lst = unknownDict[pName]
            if len(lst) == 1: unknownDict[pName] = lst[0]
        return unknownDict
