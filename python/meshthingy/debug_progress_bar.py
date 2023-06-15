def _get_progress_string(progress: float, length: int = 49, blinked = False) -> str:
    big_block = "█"
    empty_block = "░"
    half_block = "▒"
    
    num_blocks = int(round(progress * length))
    num_empty = length - num_blocks
    if not blinked:
        return big_block * num_blocks + empty_block * num_empty
    if num_blocks == length: # so that there is no extra half block if the progress is 100%
        return big_block * num_blocks
    return big_block * num_blocks + half_block + empty_block * (num_empty - 1)    