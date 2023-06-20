from run import run

_filename = "runtime_data_2.csv"

with open(f"{_filename}", "w") as f:
    f.write("side_len,a_avg, b_avg, w_avg, a_clr, b_clr, w_clr\n")
    for side_len in range(10, 101, 10):
        print(f"Running for side length {side_len}...")
        averaging_times, coloring_times = run((side_len, side_len), 100, 10)
        
        avg_avg = averaging_times[0]
        best_avg = averaging_times[1]
        worst_avg = averaging_times[2]
        
        avg_clr = coloring_times[0]
        best_clr = coloring_times[1]
        worst_clr = coloring_times[2]        
        
        f.write(f"{side_len},{avg_avg},{best_avg},{worst_avg},{avg_clr},{best_clr},{worst_clr}\n")
    
    # close and save the file
    f.close()