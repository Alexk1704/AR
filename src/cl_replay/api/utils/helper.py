def cast_args(arg):
    ''' casts an str or the content of a list arg to an int/list containing ints, returns None if not castable '''
    _arg = None
    if      isinstance(arg, str):   _arg = int(arg)
    elif    isinstance(arg, list):  _arg = [cast_args(v) for v in arg]
    return _arg


def wrap_in_arr(arg):
    ''' wraps an integer or float into a list '''
    _arg = None
    if isinstance(arg, int) or isinstance(arg, float) or isinstance(arg, str): _arg = [arg]
    else: _arg = arg
    return _arg


def target_ref(targets, model_layers):
    ''' returns the layer references/target layer(s) for an integer or a list of integers '''
    target = None
    if      isinstance(targets, list):    target = [model_layers.get(idx) for idx in targets]
    elif    isinstance(targets, int):     target = model_layers.get(targets)
    return target
