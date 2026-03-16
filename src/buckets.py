from collections import deque
import math

class Buckets:
    """Maintains edge counts in (slide-sized) time buckets for a sliding window."""
    def __init__(self, base_time, slide):
        self.base_time = base_time
        self.slide = slide
        self.buckets = deque()
    
    def addEdge(self, start_time):
        if not self.buckets or start_time >= self.buckets[-1][0] + self.slide:
            self.buckets.append((self.base_time + (start_time - self.base_time) // self.slide * self.slide, 1))
        else:
            self.buckets[-1] = (self.buckets[-1][0], self.buckets[-1][1] + 1)
    
    def getCount(self, time):
        """Returns the count of edges in the bucket corresponding to `time`. In range [time, time + slide).
        """
        for idx in range(len(self.buckets)):
            if self.buckets[idx][0] <= time and time < self.buckets[idx][0] + self.slide:
                print(f"Bucket found for time {time}: {self.buckets[idx][0]} to {self.buckets[idx][0] + self.slide}, count: {self.buckets[idx][1]}")
                return self.buckets[idx][1]
        return None

    def removeBefore(self, time):
        while self.buckets and self.buckets[0][0] + self.slide <= time:
            self.buckets.popleft()
    
    def shed(self, time):
        