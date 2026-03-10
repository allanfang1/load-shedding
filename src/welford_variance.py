
class WelfordVariance:
    def __init__(self):  
        self.mean = 0.0   
        self.count = 0   
        self.M2 = 0.0 

    def add_var(self, x: float):
        self.count += 1
        old_mean = self.mean
        self.mean += (x - self.mean) / self.count
        self.M2 += (x - old_mean) * (x - self.mean)

    def remove_var(self, x: float):
        if self.count == 0:
            raise ValueError("No data points to remove.")
        self.count -= 1
        new_mean = self.mean
        self.mean -= (x - self.mean) / self.count
        self.M2 -= (x - new_mean) * (x - self.mean)
        
    def get_mean(self) -> float:
        if self.count == 0:
            raise ValueError("No data points to calculate mean.")
        return self.mean

    def get_variance(self) -> float:
        if self.count == 0:
            raise ValueError("No data points to calculate variance.")
        return self.M2 / self.count
    
    def get_sample_variance(self) -> float: # not relevant for our use case
        return self.M2 / (self.count - 1)