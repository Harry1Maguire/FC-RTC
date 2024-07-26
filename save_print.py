def save_print_output(file_path):
    import sys
    class Logger(object):
        def __init__(self, file):
            self.terminal = sys.stdout
            self.log = file
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
        def flush(self):
            pass

    with open(file_path, 'w') as f:
        sys.stdout = Logger(f)