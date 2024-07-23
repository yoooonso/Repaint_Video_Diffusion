import argparse
import os
import glob
import numpy as np
import torch
from PIL import Image
import skimage.io
import cv2
import sys

sys.path.append('./Repaint_SVD/models')

from diffusers.utils import load_image
from Repaint_SVD.utils.Warper import Warper
from Repaint_SVD.pipelines.pipeline_stable_video_diffusion import StableVideoDiffusionPipeline, export_to_gif, export_images_to_mp4

def erode_mask(mask, erode_radius=5):
    # Ensure the image is in grayscale (binary: 0 or 255)
    mask_array = np.array(mask.convert('L'))
    
    # Define the erosion kernel size
    kernel = np.ones((erode_radius, erode_radius), np.uint8)
    
    # Erode the image
    eroded_image = cv2.erode(mask_array, kernel, iterations=1)
    
    # Apply threshold to ensure the image remains binary
    _, binary_image = cv2.threshold(eroded_image, 254, 255, 0)
    
    # Convert back to PIL Image for consistency with the rest of your pipeline
    pil_image = Image.fromarray(binary_image)
    return pil_image

def load_and_erode_mask(image_path, erode_radius=5):
    # Load the image as a grayscale image (binary: 0 or 255)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Define the erosion kernel size
    kernel = np.ones((erode_radius, erode_radius), np.uint8)
    
    # Erode the image
    eroded_image = cv2.erode(image, kernel, iterations=1)
    
    # Apply threshold to ensure the image remains binary
    _, binary_image = cv2.threshold(eroded_image, 254, 255, 0)
    
    # Convert back to PIL Image for consistency with the rest of your pipeline
    pil_image = Image.fromarray(binary_image)
    return pil_image

# Main pipeline execution
def main(args):
    svd_pipeline = StableVideoDiffusionPipeline.from_pretrained(
        args.ckpt, 
        torch_dtype=torch.float16,
        variant="fp16",
    ).to('cuda')

    def create_transformation_matrix(x, y):
        return np.array([
            1.0, 0.0, 0.0, x,
            0.0, 1.0, 0.0, y,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0
        ]).reshape(4, 4)

    image_path = os.path.join(args.input_dir, args.image_name)
    image = load_image(image_path).resize((args.width, args.height))
    
    
    eroded_frame_dir = os.path.join(args.output_dir, 'tmp', 'eroded_frame')
    eroded_mask_dir = os.path.join(args.output_dir, 'tmp', 'eroded_mask')
    output_image_dir = os.path.join(args.output_dir, 'output')
    pixel_erode_dir = os.path.join(args.output_dir, 'final_output')

    for subdir in [eroded_frame_dir, eroded_mask_dir, output_image_dir, pixel_erode_dir]:
        if not os.path.exists(subdir):
            os.makedirs(subdir)

    warper = Warper() 

    depth_path = os.path.join(args.input_dir, args.image_name.replace('.png', '.npz'))
    depth = np.load(depth_path, mmap_mode='r')['data'].astype(np.float32)
    depth /= 6250
    depth = np.maximum(depth, 1e-3)
    depth =  1 / depth
    depth = np.minimum(depth, 50)
    depth = cv2.resize(depth, dsize=(args.width, args.height), interpolation=cv2.INTER_LINEAR) 
    # print(depth.max(), depth.min())

    focal_x = args.focal_x
    focal_y = args.focal_y
    intrinsic = warper.camera_intrinsic_transform(args.width, args.height, focal_x, focal_y, 0.5, 0.5)

    grid_size = args.grid_size
    num_frames = grid_size * grid_size

    transformation1 = np.array([
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0
    ]).reshape(4, 4)

    camera_poses = np.zeros((1, num_frames, 4, 4))

    # warped_frame_dir = os.path.join(args.output_dir, 'frame_warped')
    # masked_frame_dir = os.path.join(args.output_dir, 'mask_warped')
    # if not os.path.exists(warped_frame_dir):
    #     os.makedirs(warped_frame_dir)
    #     os.makedirs(masked_frame_dir)


    warps = []
    masks = []

    for j in range(grid_size):
        for i in range(grid_size):
            x_offset = (i - grid_size // 2) * args.offset
            y_offset = (j - grid_size // 2) * args.offset
            transformation2 = create_transformation_matrix(-x_offset, -y_offset)
            camera_poses[0, j * grid_size + i] = transformation2

            x_prin = (i - grid_size // 2) * args.principal_move_x / args.width_ori
            y_prin = (j - grid_size // 2) * args.principal_move_y / args.height_ori
            intrinc2 = warper.camera_intrinsic_transform(args.width, args.height, focal_x, focal_y, 0.5 + x_prin, 0.5 + y_prin)

            warped_frame, mask, warped_depth, flow = warper.forward_warp(
                np.array(image), None, depth, transformation1, transformation2, intrinsic, intrinc2)
            
            # skimage.io.imsave(os.path.join(warped_frame_dir, f'frame_warped_{j+1}_{i+1}.png'), warped_frame.astype(np.uint8))
            # skimage.io.imsave(os.path.join(masked_frame_dir, f'mask_{j+1}_{i+1}.png'), skimage.img_as_ubyte(mask), check_contrast=False)
            warps.append(Image.fromarray(warped_frame.astype(np.uint8)).resize((args.width, args.height)))
            masks.append(Image.fromarray(skimage.img_as_ubyte(mask)).resize((args.width, args.height)))

    # Apply erosion to masks
    masks = [erode_mask(mask, erode_radius=args.erode_radius).resize((args.width, args.height)) for mask in masks]

    # warped_image_paths = glob.glob(os.path.join(warped_frame_dir, '*.png')) 
    # warped_image_paths.sort()
    # warps = [load_image(path).resize((args.width, args.height)) for path in warped_image_paths]

    # mask_image_paths = glob.glob(os.path.join(masked_frame_dir, '*.png'))
    # mask_image_paths.sort()
    # masks = [load_and_erode_mask(path, erode_radius=args.erode_radius).resize((args.width, args.height)) for path in mask_image_paths]

    for i, mask in enumerate(masks):
        mask.save(os.path.join(eroded_mask_dir, f'eroded_mask_{i:02d}.png'))

    masked_warps = []
    for mask, warp in zip(masks, warps):
        mask_array = np.array(mask)
        warp_array = np.array(warp)
        masked_warp = warp_array * (mask_array / 255)[:, :, None]
        masked_warps.append(Image.fromarray(np.uint8(masked_warp)))

    for i, masked_warp_img in enumerate(masked_warps):
        img_array = np.array(masked_warp_img)
        skimage.io.imsave(os.path.join(eroded_frame_dir, f'eroded_frame_{i:02d}.png'), img_array.astype(np.uint8))

    # masked_warps[0].save(os.path.join(args.output_dir, 'warped_images.gif'),
    #                      save_all=True, append_images=masked_warps[1:], optimize=False, duration=125, loop=0)

    masks = masks[:num_frames]
    warps = warps[:num_frames]

    print("Shape of camera_poses:", camera_poses.shape)

    output = svd_pipeline(
        image=image,
        height=args.height,
        width=args.width,
        num_frames=num_frames,
        num_inference_steps=args.num_inference_steps,
        motion_bucket_id=args.motion_bucket_id, 
        noise_aug_strength=args.noise_aug_strength,
        decode_chunk_size=args.decode_chunk_size,
        camera_poses=camera_poses,
        overlay_init_image=warps,
        overlay_mask_image=masks,
        overlay_end=args.overlay_end,
        generator=torch.Generator().manual_seed(111),
    ).frames[0]


    for i in range(num_frames):
        img = output[i]
        output[i] = np.array(img)
        img.save(os.path.join(output_image_dir, f'image_{i:02d}.png'))

    warped_image_paths = glob.glob(os.path.join(output_image_dir, '*.png')) 
    warped_image_paths.sort()
    results = [load_image(path).resize((args.width, args.height)) for path in warped_image_paths]

    mask_image_paths = glob.glob(os.path.join(eroded_mask_dir, '*.png'))
    mask_image_paths.sort()
    erode_masks = [load_image(path).resize((args.width, args.height)) for path in mask_image_paths]  


    for i, (mask, warp, result) in enumerate(zip(erode_masks, warps, results)):
        mask_array = np.array(mask)
        warp_array = np.array(warp)
        result_array = np.array(result)
        mask_array = cv2.GaussianBlur(mask_array, (21, 21), 0)
        masked_warp = warp_array * (mask_array / 255) + result_array * (1 - mask_array / 255)
        masked_warp = Image.fromarray(np.uint8(masked_warp))
        masked_warp.save(os.path.join(pixel_erode_dir, f'image_{i:02d}.png'))

    export_to_gif(output, os.path.join(args.output_dir, 'output.gif'), 8)
    # export_images_to_mp4(output_image_dir, os.path.join(args.output_dir, 'lightfield_i.mp4'), 8)
    export_images_to_mp4(pixel_erode_dir, os.path.join(args.output_dir, 'final_output.mp4'), 8)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--height", type=int, default=512, help="Height of the output images")
    parser.add_argument("--width", type=int, default=512, help="Width of the output images")
    parser.add_argument("--grid_size", type=int, default=5, help="Size of the grid for light field rendering")
    parser.add_argument("--ckpt", type=str, default="./pretrained_weights", help="Path to the model checkpoint")
    parser.add_argument("--input_dir", type=str, default="./input", help="Input directory containing images and depth data")
    parser.add_argument("--image_name", type=str, default="hci_1.png", help="Image name to be processed")
    parser.add_argument("--output_dir", type=str, default="./result", help="Output directory for results")
    parser.add_argument("--offset", type=float, default=0.06, help="Offset for light field rendering")
    parser.add_argument("--focal_x", type=float, default=0.1, help="Focal length for the camera")
    parser.add_argument("--focal_y", type=float, default=0.1, help="Focal length for the camera")
    parser.add_argument("--principal_move_x", type=float, default=2.0, help="Principal point move for intrinsic matrix adjustment")
    parser.add_argument("--principal_move_y", type=float, default=2.0, help="Principal point move for intrinsic matrix adjustment")
    parser.add_argument("--width_ori", type=int, default=512, help="Width of the original images for calculating principal point move")
    parser.add_argument("--height_ori", type=int, default=512, help="Height of the original images for calculating principal point move")
    parser.add_argument("--erode_radius", type=int, default=15, help="Radius for mask erosion")
    parser.add_argument("--decode_chunk_size", type=int, default=14, help="Decode chunk size")
    parser.add_argument("--num_inference_steps", type=int, default=25, help="Number of inference steps")
    parser.add_argument("--motion_bucket_id", type=int, default=127, help="Motion bucket ID")
    parser.add_argument("--noise_aug_strength", type=float, default=0.07, help="Noise augmentation strength")
    parser.add_argument("--overlay_end", type=float, default=0.7, help="Overlay end time")
    
    args = parser.parse_args()
    main(args)