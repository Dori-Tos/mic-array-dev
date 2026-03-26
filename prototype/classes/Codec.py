import numpy as np
import logging

class Codec:
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        
        pass
    
    
class G711Codec(Codec):
    def __init__(self, logger: logging.Logger):
        super().__init__(logger)


class OpusCodec(Codec):
    def __init__(self, logger: logging.Logger):
        super().__init__(logger)