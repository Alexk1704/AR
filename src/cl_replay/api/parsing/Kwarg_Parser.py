class Kwarg_Parser(object):
    ''' 
    A simple parser for Kwargs. 
        * Behaves similarly to the standard argparse module.
        * Checks the parameter added via "add_argument" if they are present in kwargs.
        * If the parser has a prefix, it will first search for the parameter with a prefix, then without one.
        * Parameter priority: 1. command line (if present) 2. kwargs 3. default value.
    '''
    def __init__(self, prefix='', external_arguments=None, verbose=False, **kwargs):
        '''
            @param prefix: search for parameters with the prefix (e.g., prefix="L3" parameter="--K" search for parameter "--L3_K")
            @param command_line_arguments: dictionary of external (e.g., command line) parameters
        '''
        self.kwargs = kwargs
        self.verbose = verbose
        if external_arguments: self.kwargs.update(external_arguments)  # overwrite kwargs with command line parameters
        self.prefix = prefix
        self.help_str = ''


    def convert(self, op, obj):
        ''' Applies op to convert the type of object. If object is a list, conversion is element-wise. '''
        if isinstance(obj, list): return list(map(op, obj))
        else:                     return op(obj)


    def get_all_parameters(self):
        ''' Return all collected arguments as a dict. '''
        return self.kwargs

    @staticmethod
    def make_list(x):
      if type(x) == type([]):
        return x
      else:
        return [x]


    def add_argument(self, arg_name, type=str, default=None, required=False, help='', choices=None, prefix=None, post_process = (lambda x: x), **kwargs):
        prefix = self.prefix if prefix is None else prefix

        # assumes 1st 2 chars are -- and ignores them
        if not arg_name.startswith('--'): raise Exception(f'argument ({arg_name}) does not start with "--" ')

        arg_name = arg_name[2:]  # remove --
        #print(f'{prefix}{arg_name}', arg_name, prefix) ;
        param_value_prio1 = self.kwargs.get(f'{prefix}{arg_name}', None)  # get value from kwargs with prefix
        param_value_prio2 = self.kwargs.get(arg_name, None)  # value from arg without prefix
        #print("Parsing", self.prefix, arg_name, prefix+arg_name)
        param_value = param_value_prio1
        if param_value is None: param_value = param_value_prio2

        if self.verbose == True: pass
            # raise Exception('unallowed "print" Exception')
        #print(f"{self.prefix}: looking for {arg_name}, found {param_value_prio1.__class__} as prio 1 and {param_value_prio2.__class__} as prio2 --> kept {param_value}, {param_value.__class__}")

        if param_value is None and required:      raise Exception(f'Invalid kwargs: {arg_name} missing!')               # if required arg is missing
        if param_value is None and not required:  param_value = default                                                 # if arg is missing use default value
        else:                                     param_value = post_process(self.convert(type, param_value))           # if arg is given apply type convert function

        if choices and param_value not in choices:  raise Exception(f'Invalid choice: {arg_name}={param_value} not in {choices}') # should choices be possible, then check if is included.
        #if arg_name not in self.kwargs:             self.kwargs[arg_name] = param_value                                 # collect all parameter in self.kwargs even the parameter is not given

        self.help_str += f'\n{help}'
        return param_value
