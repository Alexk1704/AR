from .                      import Regularizer
from cl_replay.api.utils    import log

class SingleExp(Regularizer):


    def __init__(self, tf_eps, tf_som_sigma, eps_0, somSigma_0, somSigma_inf, eps_inf, **kwargs):
        super().__init__(**kwargs)
        ''' set the internal values coming from the gmm layer '''
        self.tf_eps         = tf_eps
        self.tf_som_sigma   = tf_som_sigma
        self.eps_0          = eps_0
        self.somSigma_0     = somSigma_0
        self.somSigma_inf   = somSigma_inf
        self.eps_inf        = eps_inf

        ''' regularizer settings '''
        self.alpha          = self.parser.add_argument('--alpha',       type=float, default=self.eps_inf,        help='reaction speed (higher is slower)')
        self.gamma          = self.parser.add_argument('--gamma',       type=float, default=0.9,                help='reduction factor of somSigma')
        self.delta          = self.parser.add_argument('--delta',       type=float, default=0.05,               help='stationarity detection threshold')
        self.reset_sigma    = self.parser.add_argument('--reset_sigma', type=float, default=self.somSigma_inf,   help='reset value for sigma')  # default, reset to somSigma_inf, even if not yet completely reduced
        self.reset_eps      = self.parser.add_argument('--reset_eps',   type=float, default=self.eps_inf,        help='reset value for eps')  # default, reset to

        self.avg_long       = 0.0
        self.l0             = 0
        self.last_avg       = 0
        self.iteration      = 0
        self.W              = int(1 / self.alpha)
        self.limit          = None
        self.current_sigma  = self.somSigma_0


    def add(self, loss):
        it_in_loop      = self.iteration % self.W
        if it_in_loop   == 0: self.avg_long = 0.0
        ''' incrementally compute average loss in loop. At last iteration in loop, self.avg_long will contain exact average '''
        self.avg_long   *= it_in_loop / (it_in_loop + 1.)
        self.avg_long   += loss / (it_in_loop + 1.)
        self.iteration  += 1

        return self


    def set(self, eps=None, sigma=None):
        ''' reset the regularizer '''
        reset_eps      = eps   if eps      else self.reset_eps
        reset_sigma    = sigma if sigma    else self.reset_sigma

        reset_eps      = reset_eps    if reset_eps      > self.eps_inf       else self.eps_inf
        reset_sigma    = reset_sigma  if reset_sigma    > self.somSigma_inf  else self.somSigma_inf

        self.tf_som_sigma.assign(reset_sigma)
        self.tf_eps.assign(reset_eps)


    def _check(self):
        ''' do nothing during the first period just memorize initial loss value as a baseline '''
        if self.iteration % self.W != self.W - 1: return    # if we are not at the end of a period, do nothing

        if self.iteration // self.W == 0:  # first event: no last_avg yet, so set it trivially, no action
            self.l0         = self.avg_long
            self.last_avg   = self.l0
            return

        if self.iteration // self.W == 1:  # second event: last_avg can be set non-trivially, no action
            self.last_avg   = self.avg_long
            return

        limit = (self.avg_long - self.last_avg) / (self.last_avg - self.l0)
        self.limit = limit

        if (-2 * self.delta < limit) and (limit < self.delta):  # if energy does not increase sufficiently --> reduce!
            current_eps, self.current_sigma = [self.tf_eps.numpy(), self.tf_som_sigma.numpy()]
            current_eps *= self.gamma
            self.current_sigma *= self.gamma
            self.set(eps=current_eps, sigma=self.current_sigma)

        self.last_avg = self.avg_long  # update last_avg for next event
