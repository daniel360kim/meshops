import os
import datetime
import time
import torch

class IterativeFile:
    def __init__(self, directory, basename, extension):
        if not os.path.exists(directory):
            os.makedirs(directory)

        file_iterator = 0
        while os.path.exists(os.path.join(directory, f"{basename}{file_iterator}{extension}")):
            file_iterator += 1

        self.m_FileName = os.path.join(directory, f"{basename}{file_iterator}{extension}")

    def getFileName(self):
        return self.m_FileName


class Log:
    LOG_FILE_HEADER = "Time/Iteration, TempCalcIterTime(us), ColorCalcIterTime(us), TempCalcIterBest(us), " \
                      "ColorCalcIterBest(us), TempCalcIterWorst(us), ColorCalcIterWorst(us), TempCalcIterAvg(us), " \
                      "ColorCalcIterAvg(us)\n"

    def __init__(self, directory, length: int, width: int, num_iterations: int, fps: int, gif_hex: str):
        self.m_GPU = torch.cuda.is_available()
        self.m_CPU = not self.m_GPU
        self.m_GPU_Name = torch.cuda.get_device_name(0) if self.m_GPU else ""
        #self.m_Device_Name = os.environ.get('COMPUTERNAME', "") if self.m_CPU else ""
        self.m_GPU_Memory = torch.cuda.get_device_properties(0).total_memory if self.m_GPU else 0

        log_fs = IterativeFile(directory, "log", ".csv")
        self.m_logFile_Name = open(log_fs.getFileName(), "w+", buffering=8192)
        self.m_logFile_Name.write(self.LOG_FILE_HEADER)

        info_fs = IterativeFile(directory, "info", ".txt")
        
        import os
        print(f"\nLog file: {os.path.abspath(log_fs.getFileName())}")
        
        with open(info_fs.getFileName(), "w+") as info_file:
            info_file.write(f"Log File: {log_fs.getFileName()}\n")
            info_file.write(f"Gif File: {gif_hex}\n")
            info_file.write(f"Directory: {directory}\n")
            info_file.write("." * 30 + f"Length: {length}\n")
            info_file.write(f"Width: {width}\n")
            info_file.write(f"Iterations: {num_iterations}\n")
            info_file.write(f"FPS: {fps}\n")
            info_file.write("." * 30 + "\n")
            info_file.write(f"Date: {datetime.datetime.now()}\n")
            info_file.write(f"GPU: {self.m_GPU}\n")
            info_file.write(f"CPU: {self.m_CPU}\n")
            info_file.write(f"GPU Name: {self.m_GPU_Name}\n")
            info_file.write(f"Device Name: {self.m_Device_Name}\n")
            info_file.write(f"GPU Memory: {self.m_GPU_Memory}\n")

    def log(self, AvgTimer, ColorTimer, iteration):
        data = [
            iteration,
            f"{AvgTimer.m_Time:.3f}",
            f"{ColorTimer.m_Time:.3f}",
            f"{AvgTimer.m_BestTime:.3f}",
            f"{ColorTimer.m_BestTime:.3f}",
            f"{AvgTimer.m_WorstTime:.3f}",
            f"{ColorTimer.m_WorstTime:.3f}",
            f"{AvgTimer.m_AverageTime:.3f}",
            f"{ColorTimer.m_AverageTime:.3f}"
        ]
        self.m_logFile_Name.write(",".join(map(str, data)) + "\n")
