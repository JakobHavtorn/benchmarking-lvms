import logging


class ColoredLogsFormatter(logging.Formatter):
    """Logging colored formatter, adapted from https://stackoverflow.com/a/56944256/3638629"""

    grey = '\x1b[38;21m'
    blue = '\x1b[38;5;39m'
    yellow = '\x1b[38;5;226m'
    red = '\x1b[38;5;196m'
    bold_red = '\x1b[31;1m'
    reset = '\x1b[0m'

    def __init__(self, format: str):
        super().__init__()
        self.format_string = format

        colored_part, message_part = format.split("%(message)")
        message_part = "%(message)" + message_part

        self.FORMATS = {
            logging.DEBUG: self.grey + colored_part + self.reset + message_part,
            logging.INFO: self.blue + colored_part + self.reset + message_part,
            logging.WARNING: self.yellow + colored_part + self.reset + message_part,
            logging.ERROR: self.red + colored_part + self.reset + message_part,
            logging.CRITICAL: self.bold_red + colored_part + self.reset + message_part
        }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)
