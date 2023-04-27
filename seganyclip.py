import sys
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
import numpy as np
import open_clip
import clip
from PIL import Image
import torch
from tqdm import tqdm
import os
import math
import cv2

class SegmentAnything:
    def __init__(self, 
                 data_dir="/media/public_dataset/segany/", 
                 model_type="vit_h",
                 device='cuda'):
        data_dir = data_dir
        model_type = model_type
        checkpoint_name = {
            "default": 'sam_vit_h_4b8939.pth',
            "vit_h": 'sam_vit_h_4b8939.pth',
            "vit_l": 'sam_vit_l_0b3195.pth',
            "vit_b": 'sam_vit_b_01ec64.pth'
        }
        sam_checkpoint = checkpoint_name[model_type]
        
        sam_checkpoint = data_dir + sam_checkpoint

        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.sam.to(device=device)

        self.predictor = SamPredictor(self.sam)
        self.device = device
        
    def set_image(self, image):
        self.image = image
        self.predictor.set_image(image)
    
    def predict_object_coord(self, coord, without_mask=False):
        # predict objects by the input coordinate (x, y)
        # return 3 or less object predicted by Seg-Anything with their scores
        # mask out pixels that are not related to objects if 'without_mask'==False
        # crop rectangles related to objects if 'without_mask'==True
        input_point = np.array([[coord[0], coord[1]]])
        input_label = np.array([1])
        masks, scores, logits = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )
        
        objects = []
        object_scores = []
        for i, (mask, score) in enumerate(zip(masks, scores)):
            if not mask.any():
                continue
            image_object = self.image.copy()
            if not without_mask:
                image_object[np.logical_not(mask)] = (255,255,255)
            xmin, ymin, xmax, ymax = self.from_mask_to_bbox(mask)
            image_object = image_object[ymin:ymax+1, xmin:xmax+1]
            objects.append(image_object)
            object_scores.append(score)
            
        return objects, object_scores  
    
    def crop_multires_around_coord(self, coord, down=2, res=3):
        # crop rectangles in multi-resolutions around the input coordinate (x, y)

        H, W, _ = self.image.shape

        assert(down > 0 and res > 0)
        
        objects = []
        H_, W_ = H//down, W//down
        for _ in range(res):
            H_, W_ = H_//down, W_//down
            image_object = self.image.copy()
            xmin, ymin, xmax, ymax = max(coord[0]-W_, 0), max(coord[1]-H_, 0), min(coord[0]+W_, W), min(coord[1]+H_, H)
            image_object = image_object[ymin:ymax, xmin:xmax]
            objects.append(image_object)
            
        return objects 
    
    @staticmethod
    def from_mask_to_bbox(mask):
        mask_indices = np.where(mask)
        xmin, ymin, xmax, ymax = min(mask_indices[1]), min(mask_indices[0]), max(mask_indices[1]), max(mask_indices[0])
        return xmin, ymin, xmax, ymax
    
    @staticmethod
    def make_per_pixel_point_prompt(image_size):
        # image_size: H x W
        x = np.arange(image_size[1])
        y = np.arange(image_size[0])
        xv, yv = np.meshgrid(x, y)

        points = np.stack([yv, xv], axis=-1).reshape(-1, 1, 2)
        labels = np.ones(image_size[0]*image_size[1]).reshape(-1, 1)
        return points, labels


class AutoSegmentAnything:
    def __init__(self, 
                 data_dir="./checkpoints/", 
                 model_type="vit_h",
                 device='cuda'):
        data_dir = data_dir
        model_type = model_type
        checkpoint_name = {
            "default": 'sam_vit_h_4b8939.pth',
            "vit_h": 'sam_vit_h_4b8939.pth',
            "vit_l": 'sam_vit_l_0b3195.pth',
            "vit_b": 'sam_vit_b_01ec64.pth'
        }
        sam_checkpoint = checkpoint_name[model_type]
        
        sam_checkpoint = os.path.join(data_dir, sam_checkpoint)

        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.sam.to(device=device)

        self.generator = SamAutomaticMaskGenerator(
            model=self.sam,
            points_per_side = 32,
            points_per_batch = 64,
            pred_iou_thresh = 0.86,
            stability_score_thresh = 0.92,
            stability_score_offset = 1.0,
            box_nms_thresh = 0.75,
            crop_n_layers = 2,
            crop_nms_thresh = 0.75,
            crop_overlap_ratio = 0.66,
            crop_n_points_downscale_factor = 2,
            min_mask_region_area = 100
        )

        # self.generator = SamAutomaticMaskGenerator(
        #     model=self.sam,
        #     points_per_side=64,
        #     pred_iou_thresh=0.8,
        #     stability_score_thresh=0.8,
        #     crop_n_layers=0,
        #     crop_n_points_downscale_factor=0,
        #     min_mask_region_area=100,  # Requires open-cv to run post-processing
        # )

        self.device = device

    def set_image(self, image):
        self.image = image
        
    def generate_masks(self):
        return self.generator.generate(self.image)

class CLIP:
    def __init__(self, similarity_scale=10):
        self.similarity_scale = similarity_scale

    def set_similarity_scale(self, similarity_scale):
        self.similarity_scale = similarity_scale

    def encode_image(self, image):
        image = self.preprocess(Image.fromarray(image)).unsqueeze(0).to(device=self.device).half()
        with torch.no_grad():
            image_features = self.model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features
    
    def encode_text(self, text_list):
        text = self.tokenizer(text_list).to(device=self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features

    def encode_text_with_prompt_ensemble(self, text_list, prompt_templates=None):
        # using default prompt templates for ImageNet
        if prompt_templates == None:
            # prompt_templates = ['a bad photo of a {}.', 'a photo of many {}.', 'a sculpture of a {}.', 'a photo of the hard to see {}.', 'a low resolution photo of the {}.', 'a rendering of a {}.', 'graffiti of a {}.', 'a bad photo of the {}.', 'a cropped photo of the {}.', 'a tattoo of a {}.', 'the embroidered {}.', 'a photo of a hard to see {}.', 'a bright photo of a {}.', 'a photo of a clean {}.', 'a photo of a dirty {}.', 'a dark photo of the {}.', 'a drawing of a {}.', 'a photo of my {}.', 'the plastic {}.', 'a photo of the cool {}.', 'a close-up photo of a {}.', 'a black and white photo of the {}.', 'a painting of the {}.', 'a painting of a {}.', 'a pixelated photo of the {}.', 'a sculpture of the {}.', 'a bright photo of the {}.', 'a cropped photo of a {}.', 'a plastic {}.', 'a photo of the dirty {}.', 'a jpeg corrupted photo of a {}.', 'a blurry photo of the {}.', 'a photo of the {}.', 'a good photo of the {}.', 'a rendering of the {}.', 'a {} in a video game.', 'a photo of one {}.', 'a doodle of a {}.', 'a close-up photo of the {}.', 'a photo of a {}.', 'the origami {}.', 'the {} in a video game.', 'a sketch of a {}.', 'a doodle of the {}.', 'a origami {}.', 'a low resolution photo of a {}.', 'the toy {}.', 'a rendition of the {}.', 'a photo of the clean {}.', 'a photo of a large {}.', 'a rendition of a {}.', 'a photo of a nice {}.', 'a photo of a weird {}.', 'a blurry photo of a {}.', 'a cartoon {}.', 'art of a {}.', 'a sketch of the {}.', 'a embroidered {}.', 'a pixelated photo of a {}.', 'itap of the {}.', 'a jpeg corrupted photo of the {}.', 'a good photo of a {}.', 'a plushie {}.', 'a photo of the nice {}.', 'a photo of the small {}.', 'a photo of the weird {}.', 'the cartoon {}.', 'art of the {}.', 'a drawing of the {}.', 'a photo of the large {}.', 'a black and white photo of a {}.', 'the plushie {}.', 'a dark photo of a {}.', 'itap of a {}.', 'graffiti of the {}.', 'a toy {}.', 'itap of my {}.', 'a photo of a cool {}.', 'a photo of a small {}.', 'a tattoo of the {}.', 'there is a {} in the scene.', 'there is the {} in the scene.', 'this is a {} in the scene.', 'this is the {} in the scene.', 'this is one {} in the scene.']
            # easier ones
            prompt_templates = ['a photo of a {}.', 'This is a photo of a {}', 'This is a photo of a small {}', 'This is a photo of a medium {}', 'This is a photo of a large {}', 'This is a photo of a {}', 'This is a photo of a small {}', 'This is a photo of a medium {}', 'This is a photo of a large {}', 'a photo of a {} in the scene', 'a photo of a {} in the scene', 'There is a {} in the scene', 'There is the {} in the scene', 'This is a {} in the scene', 'This is the {} in the scene', 'This is one {} in the scene']

        with torch.no_grad():
            text_features = []
            for t in text_list:
                prompted_t = [template.format(t) for template in prompt_templates]
                class_embeddings = self.encode_text(prompted_t)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                text_features.append(class_embedding)
            text_features = torch.stack(text_features, dim=0)

        return text_features
    
    def predict_similarity_objects_with_feature_attention_batch(self, image_features, text, prompt_ensemble=False, batch_size=1024, top_k=1, threshold=0., projection=None):

        # use prompt templetes to prompt input texts
        text_features = self.encode_text(text) if not prompt_ensemble else self.encode_text_with_prompt_ensemble(text)
        if projection is not None:
            text_features = projection(text_features.float()).half()
        
        image_shape = image_features.shape[:2]
        
        batches = self.separate_image_features_batches(image_features, batch_size)
        batches_similarity = [[] for _ in range(top_k)]
        
        for image_features in batches:
            # don't need  fuse features if only one dim.
            if image_features.shape[-2] != 1:
                feature_similarity = self.similarity_scale * (image_features @ text_features.T)
                feature_similarity = torch.moveaxis(feature_similarity, -1, 0).softmax(axis=-1)

                image_features = (image_features[None,...] * feature_similarity[...,None]).sum(axis=-2)
                # overall similarity of objects (that detected from a pixel) with the text prompt
                similarity = self.similarity_scale * (image_features @ text_features.T)
                similarity = torch.stack([similarity[i, ..., i] for i in range(len(similarity))])
            else:
                similarity = torch.moveaxis((image_features @ text_features.T), -1, 0).squeeze(-1)
                
            for i in range(top_k):
                similarity_max = similarity.max(0).values
                similarity_argmax = similarity.argmax(0)
                similarity[similarity_argmax] = -1
                similarity_argmax[similarity_max <= threshold] = -1
                batches_similarity[i].append(similarity_argmax.cpu())
            
        similarity_argmax = [self.merge_image_features_batches(batches_similarity[i], image_shape) for i in range(top_k)]

        # similarity (len(text), len(text)): similarity scores when each text as input (input_text_for_attention, H, W, relation_with_each_text)
        return similarity_argmax
    
    @staticmethod
    def separate_image_features_batches(image_features, batch_size=1024):
        H, W = image_features.shape[:2]
        image_features = image_features.reshape(H*W, *image_features.shape[2:])
        batch_indices = []
        idx = 0
        while idx < H*W:
            batch_indices.append((idx, min(idx+batch_size, H*W)))
            idx += batch_size
            
        batches = []
        for start, end in batch_indices:
            batches.append(image_features[start:end])
        
        return batches
    
    @staticmethod 
    def merge_image_features_batches(batches, image_shape):
        H, W = image_shape
        image_features = torch.cat(batches, axis=0)
        
        return image_features.reshape(H, W, *image_features.shape[1:])


class OpenAICLIP(CLIP):
    def __init__(self, 
                 model_type='ViT-B/16', 
                 device='cuda'):
        super().__init__()

        self.model, self.preprocess = clip.load(model_type)
        self.model.to(device=device)
        self.tokenizer = clip.tokenize
        self.device = device
    
class OpenCLIP(CLIP):
    def __init__(self, 
                 model_type='ViT-B-16', 
                 pretrained='laion2b_s34b_b88k', 
                 device='cuda'):
        super().__init__()

        self.model, _, self.preprocess = open_clip.create_model_and_transforms('ViT-B-16', pretrained=pretrained, precision="fp16")
        self.model.to(device=device)
        self.tokenizer = open_clip.get_tokenizer(model_type)
        self.device = device


class SegAnyCLIP:
    def __init__(self, 
                 segany, 
                 clip, 
                 device='cuda'):
        self.segany = segany
        self.clip = clip
        self.clip_n_dims = 512
        self.n_objs = 3
        self.zeros = torch.zeros((1, self.clip_n_dims), device=device).half()
        self.device = device
    
    def encode_image(self, without_mask=False):
        # predect per-pixels features for 'image' in 'SegmentAnything' based on predicted objects
        # mask out pixels that are not related to objects if 'without_mask'==False
        # crop rectangles related to objects if 'without_mask'==True
        # output shape: (H, W, clip_n_dims, n_objs)

        image = self.segany.image
        coords = self.make_per_pixel_point_prompt(image.shape)
        
        H, W, _ = image.shape
        
        image_pixel_embeddings = []
        
        with torch.no_grad():
            for coord in tqdm(coords):
                objects, _ = self.segany.predict_object_coord(coord, without_mask=without_mask)
                pixel_embeddings = []
                for single_object in objects:
                    pixel_embeddings.append(self.clip.encode_image(single_object))
                for _ in range(self.n_objs-len(objects)):  
                    pixel_embeddings.append(self.zeros)
                pixel_embeddings = torch.cat(pixel_embeddings, dim=0)
                image_pixel_embeddings.append(pixel_embeddings)
                
        image_pixel_embeddings = torch.cat(image_pixel_embeddings, axis=0).reshape(H, W, self.n_objs, self.clip_n_dims)   
        return image_pixel_embeddings 
    
    def encode_image_multires_crop(self):
        # predect per-pixels features for 'image' in 'SegmentAnything' by cropping
        # crop rectangles related to objects if 'without_mask'==True
        # output shape: (H, W, clip_n_dims, n_objs)

        image = self.segany.image
        coords = self.make_per_pixel_point_prompt(image.shape)
        
        H, W, _ = image.shape
        
        image_pixel_embeddings = []
        
        for coord in tqdm(coords):
            objects = self.segany.crop_multires_around_coord(coord)
            pixel_embeddings = []
            for single_object in objects:
                pixel_embeddings.append(self.clip.encode_image(single_object))
            for _ in range(self.n_objs-len(objects)):  
                pixel_embeddings.append(self.zeros)
            pixel_embeddings = torch.cat(pixel_embeddings, dim=0)
            image_pixel_embeddings.append(pixel_embeddings)
                
        image_pixel_embeddings = torch.cat(image_pixel_embeddings, axis=0).reshape(H, W, self.n_objs, self.clip_n_dims)   
        return image_pixel_embeddings 

    @staticmethod       
    def make_per_pixel_point_prompt(image_size):
        # image_size: H x W
        x = np.arange(image_size[1])
        y = np.arange(image_size[0])
        xv, yv = np.meshgrid(x, y)
        points = np.stack([xv, yv], axis=-1).reshape(-1, 2)
        return points
        

class AutoSegAnyCLIP:
    def __init__(self, 
                 segany, 
                 clip, 
                 device='cuda'):
        self.segany = segany
        self.clip = clip
        self.clip_n_dims = 512
        self.n_objs = 3
        self.zeros = torch.zeros((1, self.clip_n_dims), device=device).half()
        self.device = device
    
    def encode_image(self, bbox_crop=False, extent_segmentation_mask=0, blur=False):
        # predect per-pixels features for 'image' in 'SegmentAnything' based on predicted objects
        # mask out pixels that are not related to objects if 'bbox_crop'==False
        # crop rectangles related to objects if 'bbox_crop'==True
        # extent_segmentation_mask: extent pixels of an area from each segmentation mask for bigger coverage
        # output shape: (H, W, clip_n_dims, n_objs)

        image = self.segany.image
        
        H, W, _ = image.shape
        
        image_pixel_embeddings = []

        masks = self.segany.generate_masks()

        check_mask_covered = torch.zeros(image.shape[:2])

        for i, mask in enumerate(masks):
            masks[i]['segmentation'] = self.segmentmap_extent_multi(mask['segmentation'], extent_segmentation_mask)
            check_mask_covered[masks[i]['segmentation']] = 1

        point_to_mask = {}
        for y in range(H):
            for x in range(W):
                point_to_mask[(x, y)] = []
        for i, mask in enumerate(masks):
            ys, xs = np.where(mask['segmentation'])
            for x, y in zip(xs, ys):
                point_to_mask[(x, y)] += [i]

        objects = []
        object_scores = []
        object_areas = []

        background_color = np.array([255.,255.,255.])*0.

        for i, mask in enumerate(masks):
            image_object = image.copy().astype('float')
            if bbox_crop:
                if blur:
                    image_blur = cv2.GaussianBlur(image_object, (5, 5), 0)
                    image_object[np.logical_not(mask['segmentation'])] = image_blur[np.logical_not(mask['segmentation'])]
                image_object[np.logical_not(mask['segmentation'])] *= 0.75
                image_object[np.logical_not(mask['segmentation'])] += background_color * 0.25
            else:
                image_object[np.logical_not(mask['segmentation'])] = background_color
            image_object = image_object.astype('uint8')
            xmin, ymin, xmax, ymax = self.from_mask_to_bbox(mask['segmentation'], extent=0.01)
            image_object = image_object[ymin:ymax+1, xmin:xmax+1]
            objects.append(image_object)
            object_scores.append(mask['predicted_iou'])
            object_areas.append(mask['area'])
        
        for point in point_to_mask.keys():
            point_to_mask[point] = sorted(point_to_mask[point], key=lambda x: (object_areas[x], object_scores[x]), reverse=True)

        objects_embeddings = []
        for single_object in objects:
            objects_embeddings.append(self.clip.encode_image(single_object))

        # self.zeros = self.clip.encode_image(image)
        # with torch.no_grad():
        #     image_crop = image.copy()
        #     mask_covered = (check_mask_covered==1)
        #     mask_not_covered = (check_mask_covered==0)
        #     image_crop[mask_covered] = (0, 0, 0)
        #     if mask_not_covered.any():
        #         xmin, ymin, xmax, ymax = self.from_mask_to_bbox(mask_not_covered)
        #         image_crop = image_crop[ymin:ymax+1, xmin:xmax+1]
        #     self.zeros = self.clip.encode_image(image_crop)
            
        image_pixel_embeddings = []
        for y in range(H):
            for x in range(W):
                pixel_embeddings = [objects_embeddings[object_id] for object_id in point_to_mask[(x, y)][:self.n_objs]]
                for i in range(self.n_objs-len(pixel_embeddings)):
                    pixel_embeddings.append(self.zeros)
                image_pixel_embeddings.append(torch.cat(pixel_embeddings, axis=0))
        image_pixel_embeddings = torch.cat(image_pixel_embeddings, axis=0).reshape(H, W, self.n_objs, self.clip_n_dims)

        return image_pixel_embeddings
    
    def encode_image_concept_fusion(self, bbox_crop=False, extent_segmentation_mask=0):
        # predect per-pixels features for 'image' in 'SegmentAnything' based on predicted objects
        # mask out pixels that are not related to objects if 'bbox_crop'==False
        # crop rectangles related to objects if 'bbox_crop'==True
        # extent_segmentation_mask: extent pixels of area from each segmentation mask for bigger coverage
        # output shape: (H, W, clip_n_dims, n_objs)

        image = self.segany.image
        
        H, W, _ = image.shape
        
        image_pixel_embeddings = []

        masks = self.segany.generate_masks()

        for i, mask in enumerate(masks):
            masks[i]['segmentation'] = self.segmentmap_extent_multi(mask['segmentation'], extent_segmentation_mask)

        point_to_mask = {}
        for y in range(H):
            for x in range(W):
                point_to_mask[(x, y)] = []
        for i, mask in enumerate(masks):
            ys, xs = np.where(mask['segmentation'])
            for x, y in zip(xs, ys):
                point_to_mask[(x, y)] += [i]

        objects = []
        object_scores = []
        object_areas = []
        for i, mask in enumerate(masks):
            image_object = image.copy()
            if not bbox_crop:
                image_object[np.logical_not(mask['segmentation'])] = (255,255,255)
            xmin, ymin, xmax, ymax = self.from_mask_to_bbox(mask['segmentation'])
            image_object = image_object[ymin:ymax+1, xmin:xmax+1]
            objects.append(image_object)
            object_scores.append(mask['predicted_iou'])
            object_areas.append(mask['area'])
        
        for point in point_to_mask.keys():
            point_to_mask[point] = sorted(point_to_mask[point], key=lambda x: (object_areas[x], object_scores[x]), reverse=True)

        objects_embeddings = []
        for single_object in objects:
            objects_embeddings.append(self.clip.encode_image(single_object))
            
        image_embeddings = self.clip.encode_image(image)
        objects_embeddings = torch.cat(objects_embeddings, axis=0)
        objects_local_global_similarity = (image_embeddings @ objects_embeddings.T).squeeze(0)
        objects_cross_similarity = (objects_embeddings @ objects_embeddings.T)
        objects_self_similarity = torch.stack([objects_cross_similarity[i, i] for i in range(len(objects_cross_similarity))])
        objects_avg_cross_similarity = ((objects_cross_similarity.sum(axis=-1) - objects_self_similarity)) / (len(objects_cross_similarity)-1)
        
        t = 1
        w_global = ((objects_local_global_similarity + objects_avg_cross_similarity)/t).softmax(-1)

        objects_embeddings_fusion = ((w_global[:, None] * image_embeddings) + ((1 - w_global[:, None])*objects_embeddings))
        objects_embeddings_fusion /= objects_embeddings_fusion.norm(dim=-1, keepdim=True)
        
        image_pixel_embeddings = []
        for y in range(H):
            for x in range(W):
                pixel_embeddings = [objects_embeddings_fusion[object_id][None, :] for object_id in point_to_mask[(x, y)][:self.n_objs]]
                for i in range(self.n_objs-len(pixel_embeddings)):
                    pixel_embeddings.append(self.zeros)
                image_pixel_embeddings.append(torch.cat(pixel_embeddings, axis=0))
        image_pixel_embeddings = torch.cat(image_pixel_embeddings, axis=0).reshape(H, W, self.n_objs, self.clip_n_dims)

        return image_pixel_embeddings

    @staticmethod
    def from_mask_to_bbox(mask, extent=0, sqrt=False):
        H, W = mask.shape[:2]
        mask_indices = np.where(mask)
        xmin, ymin, xmax, ymax = min(mask_indices[1]), min(mask_indices[0]), max(mask_indices[1]), max(mask_indices[0])
        if extent > 0:
            if sqrt:
                x_extent, y_extent = math.ceil(math.sqrt((xmax-xmin)*extent)), math.ceil(math.sqrt((ymax-ymin)*extent))
                # x_extent, y_extent = max(x_extent, y_extent), max(x_extent, y_extent)
            else:
                x_extent, y_extent = math.ceil((xmax-xmin)*extent), math.ceil((ymax-ymin)*extent)
                # x_extent, y_extent = max(x_extent, y_extent), max(x_extent, y_extent)
            xmin, ymin, xmax, ymax = max(xmin-x_extent, 0), max(ymin-y_extent, 0), min(xmax+x_extent, W-1), min(ymax+y_extent, H-1)
        return xmin, ymin, xmax, ymax

    @staticmethod
    def segmentmap_extent(segmentmap):
        segmentmap_extent = segmentmap.copy()
        H, W = segmentmap.shape
        ys, xs = np.where(segmentmap)
        for x, y in zip(xs, ys):
            extents = [(x, max(0, y-1)), (max(0, x-1), y), (min(W-1, x+1), y), (x, min(H-1, y+1))]
            for x, y in extents:
                segmentmap_extent[y, x] = True
        return segmentmap_extent

    def segmentmap_extent_multi(self, segmentmap, time=2):
        segmentmap_temp = segmentmap.copy()
        for _ in range(time):
            segmentmap_temp = self.segmentmap_extent(segmentmap_temp)
        return segmentmap_temp