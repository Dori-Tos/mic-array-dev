
class DOAEstimator:
    def __init__(self, update_rate: float = 3.0):
        self.frozen: bool = False
        self.latest_doa = None
        self.update_rate = update_rate
    
    def freeze(self):
        """
        Freeze the DOA estimator to prevent further updates.
        """
        
        self.frozen = True
    
    def unfreeze(self):
        """
        Unfreeze the DOA estimator to allow updates.
        """
        self.frozen = False
        
    def estimate_doa(self, audio_block):
        """
        Estimate the direction of arrival (DOA) from the given audio block.
        This is a placeholder method and should be overridden by subclasses with actual DOA estimation logic.
        """
        return None
    
    @property
    def is_frozen(self):
        return self.frozen
    
    
class IterativeDOAEstimator(DOAEstimator):
    def __init__(self, update_rate: float = 3.0):
        super().__init__(update_rate=update_rate)

    def estimate_doa(self, audio_block):
        self.latest_doa = 0  # Placeholder for actual DOA estimation logic
        return self.latest_doa