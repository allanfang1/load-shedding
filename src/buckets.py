from collections import deque
import math

class Buckets:
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
        for idx in range(len(self.buckets)):
            if self.buckets[idx][0] <= time and time < self.buckets[idx][0] + self.slide:
                return self.buckets[idx][1]
        return None

    def removeBefore(self, time):
        while self.buckets and self.buckets[0][0] + self.slide <= time:
            self.buckets.popleft()