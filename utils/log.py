'''
Logger
Modified from the implementation of PointGroup
'''

import logging
import os
import sys
import time

sys.path.append('../')

def create_logger(log_file):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    handler = logging.StreamHandler()
    log_format = '[%(asctime)s  %(levelname)s  %(filename)s  line %(lineno)d  %(process)d]  %(message)s'
    handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(handler)

    logging.basicConfig(level=logging.INFO, format=log_format, filename=log_file)    # filename: build a FileHandler
    return logger

def init(Method, split='train'):
    log_file = os.path.join(
        'exp/' + Method + '/' + split + '-{}.log'.format(time.strftime("%Y%m%d_%H%M%S", time.localtime()))
    )
    if not os.path.exists(os.path.dirname(log_file)):
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = create_logger(log_file)
    logger.info('************************ Start Logging ************************')
    return logger