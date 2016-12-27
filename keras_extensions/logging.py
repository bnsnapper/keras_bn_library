from __future__ import division
import sys
import os

# XXX: also redirect C streams, for capturing nvcc messages (else it looks like there are excessive newlines in log file)

class ConsoleAndFileLogger(object):
    """
    Logging helper which prints to a file and a stream (e.g. stdout) simultaneously.
    Can be used to redirect standard streams (sys.stdout, sys.stderr), so print(), warnings.warn(), 
    raise exceptions, etc., will be logged to console and file at the same time.

    Logging to file is done with a line buffer to allow updating lines for things like progress bars.
    """
    def __init__(self, filename='logfile.log', mode='w', stream=sys.stdout):
        assert mode in ['w', 'a']
        assert hasattr(stream, 'write') and hasattr(stream, 'flush') # basic check for valid stream, because redirecting sys.stdout,stderr to invalid Logger can cause trace not to be printed
        self.stream = stream
        self.file = open(filename, mode)
        self.linebuf = ''

    def __del__(self):
        # flush remainder of line buffer, and close file
        try:
            if len(self.linebuf) > 0:
                self.file.write(self.linebuf)
                self.file.flush()
            self.file.close()
        except:
            pass # file may be closed, unavailable, etc.

    def write(self, message):
        # write to stream (e.g. stdout)
        try:
            self.stream.write(message)
            self.stream.flush()
        except:
            pass # stream may be closed, unavailable, etc.

        # write to file (using line buffer to avoid writing many lines for things 
        # like progress bars that are erased and updated using '\b' and/or '\r')
        for c in message:
            if c == '\b':
                self.linebuf = self.linebuf[:-1]
            elif c == '\n':
                self.linebuf += c
                if len(self.linebuf) > 0:
                    try:
                        self.file.write(self.linebuf)
                        self.file.flush()
                    except:
                        pass # file may be closed, unavailable, etc.
                self.linebuf = ''
            elif c == '\r':
                self.linebuf = ''
            else:
                self.linebuf += c

    def flush(self):
        pass # already flushes each write

    #def __getattr__(self, attr):
    #    # all other attributes from stream
    #    # for code which assumes sys.stdout is a full fledged file object with 
    #    # methods such as fileno() (which includes code in the python standard library)
    #    return getattr(self.stream, attr)

def _mkdir(path, mode=0o777):
    if path == '':
        return
    try:
        os.makedirs(path, mode)
    except OSError as exc:
        import errno
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass # ignore exception because folder already exists
        else:
            raise # re-raise other exceptions
        
class log_to_file(object):
    """
    Log to file decorator.

    Example
    -------
    >>> @log_to_file('main.log', mode='a')
    >>> def main():
    >>>    print('hello world!')
    """
    def __init__(self, filename, mode='w', stream=sys.stdout):
        self.filename = filename
        self.mode = mode
        self.stream = stream

    def __call__(self, original_func):
        decorator_self = self
        def wrapped(*args, **kwargs):
            # keep track of original stdout, stderr
            old_stdout = sys.stdout
            old_stderr = sys.stderr

            # ensure path for log file exists (we do this here becaues typically main() is wrapped, 
            # so there is not opportunity to create folder before)
            _mkdir(os.path.dirname(self.filename))

            # create simultaneous stream and file logger
            logger = ConsoleAndFileLogger(self.filename, self.mode, self.stream)
    
            # flush stdout, stderr
            sys.stdout.flush()
            sys.stderr.flush()
    
            # redirect stdout, stderr
            sys.stdout = logger
            sys.stderr = logger

            # call original function
            original_func(*args, **kwargs)

            # restore original stdout, stderr
            sys.stdout = old_stdout
            sys.stderr = old_stderr
        return wrapped

