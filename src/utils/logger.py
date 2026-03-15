from config import LOG_PATH
import logging

class WriteLogs():

    # Logs simple messages with date and time showing key info like fps and audio command
    
    def __init__(self):
        logging.basicConfig(
            filename=LOG_PATH,       
            level=logging.INFO,           
            format='%(asctime)s | %(levelname)s | %(message)s'
        )
        
    def write_log(self, message):
        logging.info(message)

