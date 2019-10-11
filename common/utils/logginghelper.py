import logging
import sys

# has to be defined before adding other logging handler otherwise not working
logging.basicConfig(
    format='%(message)s',
    stream=sys.stdout,
    level=logging.INFO)


def setup_file_logging(log_file: str):
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(filename)s:%(funcName)s %(levelname)s: %(message)s'))
    logging.getLogger().addHandler(file_handler)
    logging.info('Set up logging. Log file: {}'.format(log_file))
