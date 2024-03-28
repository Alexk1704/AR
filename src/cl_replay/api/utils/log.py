import logging


class CustomFormatter(logging.Formatter):

    """
    for i in range(30, 37 + 1):
        print("\033[%dm%d\t\t\033[%dm%d" % (i, i, i + 60, i + 60))

    print("\033[39m\\033[49m                 - Reset color")
    print("\\033[2K                          - Clear Line")
    print("\\033[<L>;<C>H or \\033[<L>;<C>f  - Put the cursor at line L and column C.")
    print("\\033[<N>A                        - Move the cursor up N lines")
    print("\\033[<N>B                        - Move the cursor down N lines")
    print("\\033[<N>C                        - Move the cursor forward N columns")
    print("\\033[<N>D                        - Move the cursor backward N columns\n")
    print("\\033[2J                          - Clear the screen, move to (0,0)")
    print("\\033[K                           - Erase to end of line")
    print("\\033[s                           - Save cursor position")
    print("\\033[u                           - Restore cursor position\n")
    print("\\033[4m                          - Underline on")
    print("\\033[24m                         - Underline off\n")
    print("\\033[1m                          - Bold on")
    print("\\033[21m                         - Bold off")
    """

    #4-bit colours: [FG;BGm
    white =     "\x1b[97;20m"
    yellow =    "\x1b[93;20m"
    red =       "\x1b[91;20m"
    reset =     "\x1b[0m"

    format = "%(asctime)s - %(levelname)-14s [%(filename)-24s:(%(lineno)4d)]: %(message)s"
    FORMATS = {
        logging.DEBUG:      yellow + format + reset,
        logging.INFO:       white + format + reset,
        logging.WARNING:    red + format + reset,
        logging.ERROR:      red + format + reset,
        logging.CRITICAL:   red + format + reset
    }

    
    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, "%H:%M:%S")
        return formatter.format(record)


log             = logging.getLogger('Experiment_Logger')# create logger
log.propagate   = False                                 # disable clash with tensorflow logger
ch              = logging.StreamHandler()               # create console handler with a higher log level

ch.setFormatter(CustomFormatter())                      # set formatting for logger
log.addHandler(ch)

CODES = {
        'DEBUG':logging.DEBUG,
        'INFO':logging.INFO,
        'WARNING':logging.WARNING,
        'ERROR':logging.ERROR,
        'CRITICAL':logging.CRITICAL
}

def change_loglevel(log_level):
    ''' 
    Change the current log level.
        @param log_level: log level (int) (see logging module)
    '''
    log_level = CODES[log_level]
    log.setLevel(log_level)
    ch.setLevel(log_level)
