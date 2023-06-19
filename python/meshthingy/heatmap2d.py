import torch
from media import ConductiveSurface, NDSquareMesh
from log import Log
from log import IterativeFile
from GIFmake import draw_gif
from timing import Timer
import numpy as np
from PIL import Image
from debug_progress_bar import _get_progress_string
import matplotlib.colors as colors
from weighted_pool import weighted_pool

#torch.set_num_threads(6)

if torch.cuda.is_available():
    print("CUDA is available, using CUDA")
    device = torch.device("cuda")
else:
    print("CUDA is not available, using CPU")
    device = torch.device("cpu")

NUM_ITERATIONS = int(input("Iteration count: "))
FPS = int(input("FPS: "))

#constants
#LENGTH = int(input("Length: "))
#WIDTH = int(input("Width: "))
#RADIUS = int(input("Radius of heating: "))
LENGTH = 50
WIDTH = 50

gifFS = IterativeFile("images/2d/", "2D", ".gif")

gif_path = gifFS.getFileName()

#Set up log
logger = Log("logs/", LENGTH, WIDTH, NUM_ITERATIONS, FPS, gif_path)
#Create average function Timer instance
PERF_average_calc_timer = Timer()
PERF_color_calc_timer = Timer()

def get_rgb_ndarr(arr: torch.Tensor) -> np.ndarray:
    scaled_vals = -0.693 * arr + 0.693
    hsv_tensor = torch.stack([scaled_vals, torch.ones_like(arr), torch.ones_like(arr)], dim=-1)
    hsv_array = hsv_tensor.cpu().numpy()

    rgb_array = colors.hsv_to_rgb(hsv_array) * 255
    frame = np.round(rgb_array).astype(np.uint8)

    return frame

time_counter = 0

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

    logger.log(PERF_average_calc_timer, PERF_color_calc_timer, time_counter) # log the timings
    print(f"{_get_progress_string(time_counter/NUM_ITERATIONS)} Iteration: {time_counter}/{NUM_ITERATIONS}", end='\r')
    

print(f"\n\nBest Averaging Time: {PERF_average_calc_timer.m_BestTime} us")
print(f"Worst Averaging Time: {PERF_average_calc_timer.m_WorstTime} us")
print(f"Average Averaging Time: {round(PERF_average_calc_timer.m_AverageTime, 2)} us")

print(f"\nBest Color Time: {PERF_color_calc_timer.m_BestTime} us")
print(f"Worst Color Time: {PERF_color_calc_timer.m_WorstTime} us")
print(f"Average Color Time: {round(PERF_color_calc_timer.m_AverageTime, 2)} us")

draw_gif(frames, gif_path, FPS)

#def runtimestep(heatmap: torch.Tensor, time_counter):
#    # Move tensors to GPU
#    heatmap = heatmap.to(device)
#    
#    # Create tensors for averaging with shifted versions of the heatmap tensor
#    shifted_left    = torch.roll(heatmap, shifts=1)
#    shifted_right   = torch.roll(heatmap, shifts=-1)
#    shifted_up      = torch.roll(heatmap, shifts=1, dims=0)
#    shifted_down    = torch.roll(heatmap, shifts=-1, dims=0)
#    
#    # Set certain edges to zero so numbers don't wrap around
#    shifted_left[:, -1] = 0
#    shifted_right[:, 0] = 0
#    shifted_up[-1, :] = 0
#    shifted_down[0, :] = 0
#    
#    # Apply padding for the edge cases
#    padded_heatmap       = torch.nn.functional.pad(heatmap, (1, 1, 1, 1), mode='constant', value=0)
#    padded_shifted_left  = torch.nn.functional.pad(shifted_left, (1, 1, 1, 1), mode='constant', value=0)
#    padded_shifted_right = torch.nn.functional.pad(shifted_right, (1, 1, 1, 1), mode='constant', value=0)
#    padded_shifted_up    = torch.nn.functional.pad(shifted_up, (1, 1, 1, 1), mode='constant', value=0)
#    padded_shifted_down  = torch.nn.functional.pad(shifted_down, (1, 1, 1, 1), mode='constant', value=0)
#    
#    # Calculate the average using tensors and avoid iteration
#    # Average the 3x3 square around each element
#    avg = (padded_heatmap + padded_shifted_left + padded_shifted_right + padded_shifted_up + padded_shifted_down) / 5
#    
#    # Remove padding
#    avg = avg[1:-1, 1:-1]
#    
#    # Move the result back to CPU if needed
#    avg = avg.cpu()
#    
#    return avg