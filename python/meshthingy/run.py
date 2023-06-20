import torch
from media import NDSquareMesh
from media_gpu import NDSquareMeshCUDA
from log import Log
from log import IterativeFile
from GIFmake import draw_gif
from timing import Timer
import numpy as np
from PIL import Image
from debug_progress_bar import _get_progress_string
import matplotlib.colors as colors

def run(size: tuple = (50, 50), iterations: int = 100, fps: int = 10) -> None:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    NUM_ITERATIONS = iterations
    FPS = fps
    LENGTH = size[0]
    WIDTH = size[1]

    gifFS = IterativeFile("images/2d/", "2D", ".gif")

    gif_path = gifFS.getFileName()

    #Set up log
    #logger = Log("logs/", LENGTH, WIDTH, NUM_ITERATIONS, FPS, gif_path)
    #Create average function Timer instance
    PERF_average_calc_timer = Timer()
    PERF_color_calc_timer = Timer()

    def get_rgb_ndarr(arr: torch.Tensor) -> np.ndarray:
        arr = arr.to(device)
        with torch.cuda.device(device):
            scaled_vals = -0.693 * arr + 0.693
            hsv_tensor = torch.stack([scaled_vals, torch.ones_like(arr), torch.ones_like(arr)], dim=-1)
            hsv_array = hsv_tensor.cpu().numpy()

            rgb_array = colors.hsv_to_rgb(hsv_array) * 255
            frame = np.round(rgb_array).astype(np.uint8)

        return frame

    time_counter = 0

    new_heatmap = ...
    if torch.cuda.is_available():
        print("[TOP LEVEL] CUDA is available, using CUDA")
        device = torch.device("cuda")
        new_heatmap = NDSquareMeshCUDA((LENGTH, WIDTH), 0)
    else:
        print("CUDA is not available, using CPU")
        device = torch.device("cpu")
        new_heatmap = NDSquareMesh((LENGTH, WIDTH), 0)
        
    new_heatmap.set_region((LENGTH//2, WIDTH//2), 15, 1)

    frames = []
    frames.append(Image.fromarray(get_rgb_ndarr(new_heatmap.get_iterable()), 'RGB'))

    for i in range(NUM_ITERATIONS):
        
        # constant heat source... add heat to the center of the mesh each iter
        # new_heatmap.set_region((LENGTH//2, WIDTH//2), RADIUS, i/NUM_ITERATIONS)
        time_counter += 1
        
        #new_heatmap.set_region((50, 50), RADIUS, 0.4)

        PERF_average_calc_timer.begin() # begin timer for average calculation
        new_heatmap.run_timestep()
        #print(new_heatmap.get_iterable()); print("\n")
        PERF_average_calc_timer.end() # end timer for average calculation
        
        #print(heatmap_random)
        PERF_color_calc_timer.begin() #begin timer for color calculation
        #print(f"shape before trying to get ndarr: {new_heatmap.get_iterable().shape}")
        frames.append(Image.fromarray(get_rgb_ndarr(new_heatmap.get_iterable()), 'RGB'))
        PERF_color_calc_timer.end() #end timer for color calculation
        print(f"{_get_progress_string(time_counter/NUM_ITERATIONS)} Iteration: {time_counter}/{NUM_ITERATIONS}", end='\r')

        #logger.log(PERF_average_calc_timer, PERF_color_calc_timer, time_counter) # log the timings
    print("\n")
    return (
        (
            round(PERF_average_calc_timer.m_AverageTime, 2),
            round(PERF_average_calc_timer.m_BestTime, 2),
            round(PERF_average_calc_timer.m_WorstTime, 2)
        ),
        (
            round(PERF_color_calc_timer.m_AverageTime, 2),
            round(PERF_color_calc_timer.m_BestTime, 2),
            round(PERF_color_calc_timer.m_WorstTime, 2)
        )
    )