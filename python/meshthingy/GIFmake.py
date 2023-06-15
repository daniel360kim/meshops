from PIL import Image, ImageDraw
import torch
  
def draw_gif(images: list, save_path: str, fps: int = 20):
    
    duration = int(round(1000 / fps)) # in milliseconds
    
    # resize all images
    for i in range(len(images)):
        images[i] = images[i].resize((500, 500), Image.NEAREST)
    
    print(f"\nSaving gif to {save_path}")

    images[0].save(
        f'{save_path}',
        save_all = True, 
        append_images = images[1:], 
        optimize = True, 
        duration = duration
    )

    
def convert_2d_tensor_to_image(tsr: torch.Tensor):
    img = Image.new('RGB', (tsr.shape[0], tsr.shape[1]), color = 'white')
    pixels = img.load()
    
    for i in range(img.size[0]):
        for j in range(img.size[1]):
            pixels[i, j] = (int(tsr[i, j] * 255), 0, 0)
    
    return img