## times each iteration, logs the best iteration, worst iteration, and average iteration time
import time
import torch

class Timer:
    m_EpochTime = 0
    m_BestTime = 0
    m_WorstTime = 0
    m_AverageTime = 0
    m_IterationCount = 0
    m_Time = 0

    def __init__(self):
        self.m_EpochTime = time.perf_counter_ns() / 1000.0


    def begin(self):
        self.m_Time = time.perf_counter_ns()
    
    def end(self):
        self.m_Time = (time.perf_counter_ns() - self.m_Time) / 1000.0
        self.m_IterationCount += 1
        self.m_AverageTime = (self.m_AverageTime + self.m_Time) / 2
        if self.m_Time > self.m_WorstTime:
            self.m_WorstTime = self.m_Time
        if self.m_Time < self.m_BestTime or self.m_BestTime == 0:
            self.m_BestTime = self.m_Time
    
    def getTimes(self):
        return (self.m_Time, self.m_BestTime, self.m_WorstTime, self.m_AverageTime)


