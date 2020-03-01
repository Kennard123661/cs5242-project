import sys
import os


class CustomLogger:
    def __init__(self, logger_file):
        self.stream = sys.stdout
        self.logger_file = logger_file
        if not os.path.exists(os.path.dirname(self.logger_file)):
            os.makedirs(os.path.dirname(self.logger_file))
        self.file_stream = open(self.logger_file, 'a')

    def write(self, data):
        self.stream.write(data)
        self.stream.flush()
        self.file_stream.write(data)
        self.file_stream.flush()

    def flush(self):
        self.stream.flush()
        self.file_stream.flush()

    def __del__(self):
        self.file_stream.close()
