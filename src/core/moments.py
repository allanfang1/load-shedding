

class Moments:
    def __init__(self):
        self.s1 = 0
        self.s2 = 0
        self.s3 = 0
        
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

    def get_mean(self, vert_count) -> float:
        return self.s1 / vert_count if vert_count > 0 else 0.0
    
    def get_variance(self, vert_count) -> float:
        if vert_count == 0:
            return 0.0
        mean = self.get_mean(vert_count)
        return (self.s2 / vert_count) - mean**2
    
    def get_skewness(self, vert_count) -> float:
        if vert_count == 0:
            return 0.0
        mean = self.get_mean(vert_count)
        var = self.get_variance(vert_count)
        if var == 0:
            return 0.0
        return (self.s3 / vert_count - 3 * mean * var - mean**3) / var**(3/2)
