import datetime
import time
import traceback
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import skimage.io
from PIL import Image

class Warper:
    def __init__(self, resolution: tuple = None):
        self.resolution = resolution
        return

    def forward_warp(self, frame1: np.ndarray, mask1: Optional[np.ndarray], depth1: np.ndarray,
                     transformation1: np.ndarray, transformation2: np.ndarray, intrinsic1: np.ndarray,
                     intrinsic2: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                                                   np.ndarray]:
        """
        Given a frame1 and global transformations transformation1 and transformation2, warps frame1 to next view using
        bilinear splatting.
        :param frame1: (h, w, 3) uint8 np array
        :param mask1: (h, w) bool np array. Wherever mask1 is False, those pixels are ignored while warping. Optional
        :param depth1: (h, w) float np array.
        :param transformation1: (4, 4) extrinsic transformation matrix of first view: [R, t; 0, 1]
        :param transformation2: (4, 4) extrinsic transformation matrix of second view: [R, t; 0, 1]
        :param intrinsic1: (3, 3) camera intrinsic matrix
        :param intrinsic2: (3, 3) camera intrinsic matrix. Optional
        """
        if self.resolution is not None:
            assert frame1.shape[:2] == self.resolution
        h, w = frame1.shape[:2]
        if mask1 is None:
            mask1 = np.ones(shape=(h, w), dtype=bool)
        if intrinsic2 is None:
            intrinsic2 = np.copy(intrinsic1)
        assert frame1.shape == (h, w, 3)
        assert mask1.shape == (h, w)
        assert depth1.shape == (h, w)
        assert transformation1.shape == (4, 4)
        assert transformation2.shape == (4, 4)
        assert intrinsic1.shape == (3, 3)
        assert intrinsic2.shape == (3, 3)

        trans_points1 = self.compute_transformed_points(depth1, transformation1, transformation2, intrinsic1,
                                                        intrinsic2)
        trans_coordinates = trans_points1[:, :, :2, 0] / trans_points1[:, :, 2:3, 0]
        trans_depth1 = trans_points1[:, :, 2, 0]

        grid = self.create_grid(h, w)
        flow12 = trans_coordinates - grid

        warped_frame2, mask2 = self.bilinear_splatting(frame1[:,:,:3], mask1, trans_depth1, flow12, None, is_image=True)
        warped_depth2 = self.bilinear_splatting(trans_depth1[:, :, None], mask1, trans_depth1, flow12, None,
                                                is_image=False)[0][:, :, 0]

        return warped_frame2, mask2, warped_depth2, flow12

    def compute_transformed_points(self, depth1: np.ndarray, transformation1: np.ndarray,
                                   transformation2: np.ndarray, intrinsic1: np.ndarray,
                                   intrinsic2: Optional[np.ndarray]):
        """
        Computes transformed position for each pixel location
        """
        if self.resolution is not None:
            assert depth1.shape == self.resolution
        h, w = depth1.shape
        if intrinsic2 is None:
            intrinsic2 = np.copy(intrinsic1)
        transformation = np.matmul(transformation2, np.linalg.inv(transformation1))

        y1d = np.array(range(h))
        x1d = np.array(range(w))
        x2d, y2d = np.meshgrid(x1d, y1d)
        ones_2d = np.ones(shape=(h, w))
        ones_4d = ones_2d[:, :, None, None]
        pos_vectors_homo = np.stack([x2d, y2d, ones_2d], axis=2)[:, :, :, None]

        intrinsic1_inv = np.linalg.inv(intrinsic1)
        intrinsic1_inv_4d = intrinsic1_inv[None, None]
        intrinsic2_4d = intrinsic2[None, None]
        depth_4d = depth1[:, :, None, None]
        trans_4d = transformation[None, None]

        unnormalized_pos = np.matmul(intrinsic1_inv_4d, pos_vectors_homo)
        world_points = depth_4d * unnormalized_pos
        world_points_homo = np.concatenate([world_points, ones_4d], axis=2)
        trans_world_homo = np.matmul(trans_4d, world_points_homo)
        trans_world = trans_world_homo[:, :, :3]
        trans_norm_points = np.matmul(intrinsic2_4d, trans_world)
        return trans_norm_points

    def bilinear_splatting(self, frame1: np.ndarray, mask1: Optional[np.ndarray], depth1: np.ndarray,
                           flow12: np.ndarray, flow12_mask: Optional[np.ndarray], is_image: bool = False) -> \
            Tuple[np.ndarray, np.ndarray]:
        """
        Using inverse bilinear interpolation based splatting
        :param frame1: (h, w, c)
        :param mask1: (h, w): True if known and False if unknown. Optional
        :param depth1: (h, w)
        :param flow12: (h, w, 2)
        :param flow12_mask: (h, w): True if valid and False if invalid. Optional
        :param is_image: If true, the return array will be clipped to be in the range [0, 255] and type-casted to uint8
        :return: warped_frame2: (h, w, c)
                 mask2: (h, w): True if known and False if unknown
        """
        if self.resolution is not None:
            assert frame1.shape[:2] == self.resolution
        h, w, c = frame1.shape
        if mask1 is None:
            mask1 = np.ones(shape=(h, w), dtype=bool)
        if flow12_mask is None:
            flow12_mask = np.ones(shape=(h, w), dtype=bool)
        grid = self.create_grid(h, w)
        trans_pos = flow12 + grid

        trans_pos_offset = trans_pos + 1
        trans_pos_floor = np.floor(trans_pos_offset).astype('int')
        trans_pos_ceil = np.ceil(trans_pos_offset).astype('int')
        trans_pos_offset[:, :, 0] = np.clip(trans_pos_offset[:, :, 0], a_min=0, a_max=w + 1)
        trans_pos_offset[:, :, 1] = np.clip(trans_pos_offset[:, :, 1], a_min=0, a_max=h + 1)
        trans_pos_floor[:, :, 0] = np.clip(trans_pos_floor[:, :, 0], a_min=0, a_max=w + 1)
        trans_pos_floor[:, :, 1] = np.clip(trans_pos_floor[:, :, 1], a_min=0, a_max=h + 1)
        trans_pos_ceil[:, :, 0] = np.clip(trans_pos_ceil[:, :, 0], a_min=0, a_max=w + 1)
        trans_pos_ceil[:, :, 1] = np.clip(trans_pos_ceil[:, :, 1], a_min=0, a_max=h + 1)

        prox_weight_nw = (1 - (trans_pos_offset[:, :, 1] - trans_pos_floor[:, :, 1])) * \
                         (1 - (trans_pos_offset[:, :, 0] - trans_pos_floor[:, :, 0]))
        prox_weight_sw = (1 - (trans_pos_ceil[:, :, 1] - trans_pos_offset[:, :, 1])) * \
                         (1 - (trans_pos_offset[:, :, 0] - trans_pos_floor[:, :, 0]))
        prox_weight_ne = (1 - (trans_pos_offset[:, :, 1] - trans_pos_floor[:, :, 1])) * \
                         (1 - (trans_pos_ceil[:, :, 0] - trans_pos_offset[:, :, 0]))
        prox_weight_se = (1 - (trans_pos_ceil[:, :, 1] - trans_pos_offset[:, :, 1])) * \
                         (1 - (trans_pos_ceil[:, :, 0] - trans_pos_offset[:, :, 0]))

        sat_depth1 = np.clip(depth1, a_min=0, a_max=1000)
        log_depth1 = np.log(1 + sat_depth1)
        depth_weights = np.exp(log_depth1 / log_depth1.max() * 50)

        weight_nw = prox_weight_nw * mask1 * flow12_mask / depth_weights
        weight_sw = prox_weight_sw * mask1 * flow12_mask / depth_weights
        weight_ne = prox_weight_ne * mask1 * flow12_mask / depth_weights
        weight_se = prox_weight_se * mask1 * flow12_mask / depth_weights

        weight_nw_3d = weight_nw[:, :, None]
        weight_sw_3d = weight_sw[:, :, None]
        weight_ne_3d = weight_ne[:, :, None]
        weight_se_3d = weight_se[:, :, None]

        warped_image = np.zeros(shape=(h + 2, w + 2, c), dtype=np.float64)
        warped_weights = np.zeros(shape=(h + 2, w + 2), dtype=np.float64)

        np.add.at(warped_image, (trans_pos_floor[:, :, 1], trans_pos_floor[:, :, 0]), frame1 * weight_nw_3d)
        np.add.at(warped_image, (trans_pos_ceil[:, :, 1], trans_pos_floor[:, :, 0]), frame1 * weight_sw_3d)
        np.add.at(warped_image, (trans_pos_floor[:, :, 1], trans_pos_ceil[:, :, 0]), frame1 * weight_ne_3d)
        np.add.at(warped_image, (trans_pos_ceil[:, :, 1], trans_pos_ceil[:, :, 0]), frame1 * weight_se_3d)

        np.add.at(warped_weights, (trans_pos_floor[:, :, 1], trans_pos_floor[:, :, 0]), weight_nw)
        np.add.at(warped_weights, (trans_pos_ceil[:, :, 1], trans_pos_floor[:, :, 0]), weight_sw)
        np.add.at(warped_weights, (trans_pos_floor[:, :, 1], trans_pos_ceil[:, :, 0]), weight_ne)
        np.add.at(warped_weights, (trans_pos_ceil[:, :, 1], trans_pos_ceil[:, :, 0]), weight_se)

        cropped_warped_image = warped_image[1:-1, 1:-1]
        cropped_weights = warped_weights[1:-1, 1:-1]

        mask = cropped_weights > 0
        with np.errstate(invalid='ignore'):
            warped_frame2 = np.where(mask[:, :, None], cropped_warped_image / cropped_weights[:, :, None], 0)

        # if is_image:
        #     assert np.min(warped_frame2) >= 0
        #     assert np.max(warped_frame2) <= 256
        #     clipped_image = np.clip(warped_frame2, a_min=0, a_max=255)
        #     warped_frame2 = np.round(clipped_image).astype('uint8')
        return warped_frame2, mask

    def bilinear_interpolation(self, frame2: np.ndarray, mask2: Optional[np.ndarray], flow12: np.ndarray,
                               flow12_mask: Optional[np.ndarray], is_image: bool = False) -> \
            Tuple[np.ndarray, np.ndarray]:
        """
        Using bilinear interpolation
        :param frame2: (h, w, c)
        :param mask2: (h, w): True if known and False if unknown. Optional
        :param flow12: (h, w, 2)
        :param flow12_mask: (h, w): True if valid and False if invalid. Optional
        :param is_image: If true, the return array will be clipped to be in the range [0, 255] and type-casted to uint8
        :return: warped_frame1: (h, w, c)
                 mask1: (h, w): True if known and False if unknown
        """
        if self.resolution is not None:
            assert frame2.shape[:2] == self.resolution
        h, w, c = frame2.shape
        if mask2 is None:
            mask2 = np.ones(shape=(h, w), dtype=bool)
        if flow12_mask is None:
            flow12_mask = np.ones(shape=(h, w), dtype=bool)
        grid = self.create_grid(h, w)
        trans_pos = flow12 + grid

        trans_pos_offset = trans_pos + 1
        trans_pos_floor = np.floor(trans_pos_offset).astype('int')
        trans_pos_ceil = np.ceil(trans_pos_offset).astype('int')
        trans_pos_offset[:, :, 0] = np.clip(trans_pos_offset[:, :, 0], a_min=0, a_max=w + 1)
        trans_pos_offset[:, :, 1] = np.clip(trans_pos_offset[:, :, 1], a_min=0, a_max=h + 1)
        trans_pos_floor[:, :, 0] = np.clip(trans_pos_floor[:, :, 0], a_min=0, a_max=w + 1)
        trans_pos_floor[:, :, 1] = np.clip(trans_pos_floor[:, :, 1], a_min=0, a_max=h + 1)
        trans_pos_ceil[:, :, 0] = np.clip(trans_pos_ceil[:, :, 0], a_min=0, a_max=w + 1)
        trans_pos_ceil[:, :, 1] = np.clip(trans_pos_ceil[:, :, 1], a_min=0, a_max=h + 1)

        prox_weight_nw = (1 - (trans_pos_offset[:, :, 1] - trans_pos_floor[:, :, 1])) * \
                         (1 - (trans_pos_offset[:, :, 0] - trans_pos_floor[:, :, 0]))
        prox_weight_sw = (1 - (trans_pos_ceil[:, :, 1] - trans_pos_offset[:, :, 1])) * \
                         (1 - (trans_pos_offset[:, :, 0] - trans_pos_floor[:, :, 0]))
        prox_weight_ne = (1 - (trans_pos_offset[:, :, 1] - trans_pos_floor[:, :, 1])) * \
                         (1 - (trans_pos_ceil[:, :, 0] - trans_pos_offset[:, :, 0]))
        prox_weight_se = (1 - (trans_pos_ceil[:, :, 1] - trans_pos_offset[:, :, 1])) * \
                         (1 - (trans_pos_ceil[:, :, 0] - trans_pos_offset[:, :, 0]))

        weight_nw = prox_weight_nw * flow12_mask
        weight_sw = prox_weight_sw * flow12_mask
        weight_ne = prox_weight_ne * flow12_mask
        weight_se = prox_weight_se * flow12_mask

        weight_nw_3d = weight_nw[:, :, None]
        weight_sw_3d = weight_sw[:, :, None]
        weight_ne_3d = weight_ne[:, :, None]
        weight_se_3d = weight_se[:, :, None]

        frame2_offset = np.pad(frame2, pad_width=((1, 1), (1, 1), (0, 0)), mode='constant', constant_values=0)
        mask2_offset = np.pad(mask2, pad_width=((1, 1), (1, 1)), mode='constant', constant_values=0)

        f2_nw = frame2_offset[trans_pos_floor[:, :, 1], trans_pos_floor[:, :, 0]]
        f2_sw = frame2_offset[trans_pos_ceil[:, :, 1], trans_pos_floor[:, :, 0]]
        f2_ne = frame2_offset[trans_pos_floor[:, :, 1], trans_pos_ceil[:, :, 0]]
        f2_se = frame2_offset[trans_pos_ceil[:, :, 1], trans_pos_ceil[:, :, 0]]

        m2_nw = mask2_offset[trans_pos_floor[:, :, 1], trans_pos_floor[:, :, 0]]
        m2_sw = mask2_offset[trans_pos_ceil[:, :, 1], trans_pos_floor[:, :, 0]]
        m2_ne = mask2_offset[trans_pos_floor[:, :, 1], trans_pos_ceil[:, :, 0]]
        m2_se = mask2_offset[trans_pos_ceil[:, :, 1], trans_pos_ceil[:, :, 0]]

        m2_nw_3d = m2_nw[:, :, None]
        m2_sw_3d = m2_sw[:, :, None]
        m2_ne_3d = m2_ne[:, :, None]
        m2_se_3d = m2_se[:, :, None]

        nr = weight_nw_3d * f2_nw * m2_nw_3d + weight_sw_3d * f2_sw * m2_sw_3d + \
             weight_ne_3d * f2_ne * m2_ne_3d + weight_se_3d * f2_se * m2_se_3d
        dr = weight_nw_3d * m2_nw_3d + weight_sw_3d * m2_sw_3d + weight_ne_3d * m2_ne_3d + weight_se_3d * m2_se_3d
        warped_frame1 = np.where(dr > 0, nr / dr, 0)
        mask1 = dr[:, :, 0] > 0

        # if is_image:
        #     assert np.min(warped_frame1) >= 0
        #     assert np.max(warped_frame1) <= 256
        #     clipped_image = np.clip(warped_frame1, a_min=0, a_max=255)
        #     warped_frame1 = np.round(clipped_image).astype('uint8')
        return warped_frame1, mask1
    
    @staticmethod
    def create_grid(h, w):
        x_1d = np.arange(0, w)[None]
        y_1d = np.arange(0, h)[:, None]
        x_2d = np.repeat(x_1d, repeats=h, axis=0)
        y_2d = np.repeat(y_1d, repeats=w, axis=1)
        grid = np.stack([x_2d, y_2d], axis=2)
        return grid
    
    @staticmethod
    def read_image(path: Path) -> np.ndarray:
        if path.suffix in ['.jpg', '.png', '.bmp']:
            image = skimage.io.imread(path.as_posix())
        elif path.suffix == '.npy':
            image = np.load(path.as_posix())
        else:
            raise RuntimeError(f'Unknown image format: {path.as_posix()}')
        return image

    # @staticmethod
    # def read_depth(path: Path) -> np.ndarray:
    #     if path.suffix == '.png':
    #         depth = skimage.io.imread(path.as_posix())
    #     elif path.suffix == '.npy':
    #         depth = np.load(path.as_posix())
    #     elif path.suffix == '.npz':
    #         with np.load(path.as_posix()) as depth_data:
    #             depth = depth_data['data']
    #     elif path.suffix == '.exr':
    #         import Imath
    #         import OpenEXR

    #         exr_file = OpenEXR.InputFile(path.as_posix())
    #         raw_bytes = exr_file.channel('B', Imath.PixelType(Imath.PixelType.FLOAT))
    #         depth_vector = np.frombuffer(raw_bytes, dtype=np.float32)
    #         height = exr_file.header()['displayWindow'].max.y + 1 - exr_file.header()['displayWindow'].min.y
    #         width = exr_file.header()['displayWindow'].max.x + 1 - exr_file.header()['displayWindow'].min.x
    #         depth = np.reshape(depth_vector, (height, width))
    #     else:
    #         raise RuntimeError(f'Unknown depth format: {path.as_posix()}')
        
    #     depth /= 6250
    #     depth = np.maximum(depth, 1e-3)
    #     depth =  1 / depth
    #     return depth

    @staticmethod
    def camera_intrinsic_transform(capture_width=512, capture_height=512,focal_x=1.25, focal_y=1.25, px=0.5, py=0.5):
        camera_intrinsics = np.eye(3)
        camera_intrinsics[0, 0] = capture_width * focal_x
        camera_intrinsics[0, 2] = capture_width*px
        camera_intrinsics[1, 1] = capture_height * focal_y
        camera_intrinsics[1, 2] = capture_height*py
        return camera_intrinsics