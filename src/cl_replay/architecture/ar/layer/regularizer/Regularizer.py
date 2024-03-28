from cl_replay.api.parsing   import Kwarg_Parser



class Regularizer_Method:
  SINGLE_EXP       = 'SingleExp'
  DEFAULT_REG      = 'DefaultReg'


class Regularizer:

    def __init__(self, **kwargs):
        prefix      = kwargs.get('prefix', None)
        self.name   = f'Regularizer_{prefix}'

        self.parser = Kwarg_Parser(**kwargs)

    def add(self, loss)                :  pass
    def set(self, eps=None, sigma=None):  pass
    def _check(self)                   :  pass
    def check_limit(self)              :  self._check()
    def __str__(self):                    return self.__class__.__name__
