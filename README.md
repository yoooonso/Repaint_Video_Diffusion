<h1 align="center">Geometrically Consistent Light Field Synthesis using Repaint Video Diffusion Model</h1>

This repository is the official implementation of [Geometrically Consistent Light Field Synthesis using Repaint Video Diffusion Model]() (*Note: The link is still being prepared*)

<p align="center"><img src="./assets/teaser.png" width="100%"></p>
> **Geometrically Consistent Light Field Synthesis using Repaint Video Diffusion Model** <br>
> Soyoung Yoon, In Kyu Park

**Abstract.** *We propose to repaint an image-to-video diffusion model to synthesize light fields that are geometrically consistent. Despite significant advancements in diffusion models for novel view synthesis, applying these models to generate a light field, i.e. fronto-parallel multiple views, has been challenging due to persistent visual and geometric consistency issues. By utilizing advances in video diffusion, we extend the temporal consistency of video diffusion to the geometric consistency of multi-view settings. By incorporating multi-view data with camera parameters, we fine-tune the image-to-video diffusion model framework for optimized multi-view diffusion. Furthermore, we propose integrating a repaint method during the sampling (denoising process) to achieve enhanced accurate camera control in multi-view diffusion, enhancing consistency by maintaining the known region in the input image. This approach enables the application of light field synthesis that requires precise camera control and demonstrates the capability of diffusion models to generate light fields with wide baselines, leveraging their unique generative power.*

## Installation
### Environment
- 64-bit Python 3.8 and PyTorch 2.2.0 or higher
- CUDA 11.8
```bash
conda env create -f environment.yaml
conda activate RepaintSVD
```

## Pretrained Weights & Data
### Pretrained Weights
You can download the pretrained weights from the following link:
[Link]() (*Note: The link is still being prepared*)
### Data Preparation
To run the inference script, you need to prepare the input data as follows:
- Place your input images in the input folder
- Ensure that each image has a corresponding depth file 
- The depth estimation for the input images was performed using the [MiDaS](https://github.com/isl-org/MiDaS) depth estimation model

```bash
input/
├── image_01.png
├── image_01.npz
└── ...
```

## Inference
### Inference
To generate Light Field images from the pretrained models, run the following command:

```
python inference_LF.py

Options:
    --height                Height of the output images (default: 512)
    --width                 Width of the output images (default: 512)
    --grid_size             Size of the grid for light field rendering (default: 5)
    --ckpt                  Path to the model checkpoint (default: "./pretrained_weights")
    --input_dir             Input directory containing images and depth data (default: "./input")
    --image_name            Image name to be processed (default: "hci_1.png")
    --output_dir            Output directory for results (default: "./result")
    --offset                Baseline for light field rendering (default: 0.06)
    --focal_x               Focal length for the camera (default: 0.1)
    --focal_y               Focal length for the camera (default: 0.1)
    --principal_move_x      Principal point move for intrinsic matrix adjustment (default: 2.0)
    --principal_move_y      Principal point move for intrinsic matrix adjustment (default: 2.0)
    --width_ori             Width of the original images for calculating principal point move (default: 512)
    --height_ori            Height of the original images for calculating principal point move (default: 512)
    --erode_radius          Radius for mask erosion (default: 15)
    --decode_chunk_size     Decode chunk size (default: 14)
    --num_inference_steps   Number of inference steps (default: 25)
    --motion_bucket_id      Motion bucket ID (default: 127)
    --noise_aug_strength    Noise augmentation strength (default: 0.07)
    --overlay_end           Overlay end time for Repaint (default: 0.7)

```

### Results

<p align="center">
  <img src="assets/example3.gif" alt="Example 3" width="250"/>
  <img src="assets/example1.gif" alt="Example 1" width="250"/>
  <img src="assets/example2.gif" alt="Example 2" width="250"/>
</p>