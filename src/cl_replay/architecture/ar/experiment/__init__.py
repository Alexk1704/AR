import sys
if not '-m' in sys.argv:
    from .Experiment_GMM import Experiment_GMM
    from .Experiment_AR import Experiment_AR