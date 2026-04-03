

class Moments:
    def __init__(self):
        self.V = 0

        self.s1 = 0
        self.s2 = 0
        self.s3 = 0

    def add_vertex(self):
        self.V += 1
        
    def increment_update(self, degree):
        """Given current degree of a vertex, to be incremented to degree + 1."""
        self.s1 += 1
        self.s2 += 2 * degree + 1
        self.s3 += 3 * degree**2 + 3 * degree + 1
    
    def decrement_update(self, degree):
        """Given current degree of a vertex, to be decremented to degree - 1."""
        self.s1 -= 1
        self.s2 -= 2 * degree - 1
        self.s3 -= 3 * degree**2 - 3 * degree + 1

    def get_mean(self) -> float:
        return self.s1 / self.V if self.V > 0 else 0.0
    
    def get_variance(self) -> float:
        if self.V == 0:
            return 0.0
        mean = self.get_mean()
        return (self.s2 / self.V) - mean**2
    
    def get_skewness(self) -> float:
        if self.V == 0:
            return 0.0
        mean = self.get_mean()
        var = self.get_variance()
        if var == 0:
            return 0.0
        return (self.s3 / self.V - 3 * mean * var - mean**3) / var**(3/2)
