import sys
from src.logger import logging

def error_message_detail(error, error_detail):
    exc_type, exc_value, exc_tb = error_detail.exc_info()
    if exc_tb is None:
        return f"Error: {str(error)}"
    
    file_name = exc_tb.tb_frame.f_code.co_filename
    return f"Error in script: [{file_name}] at line [{exc_tb.tb_lineno}] - {str(error)}"


    

class CustomException(Exception):
    def __init__(self, error_message, error_detail=sys):
        try:
            self.error_message = error_message_detail(error_message, error_detail)
        except Exception:
            self.error_message = str(error_message)
        super().__init__(self.error_message)