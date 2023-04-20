import argparse
import torch
import cv2
import os
import yaml
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from seganyclip import OpenAICLIP, OpenCLIP

device = 'cuda' if torch.cuda.is_available() else 'cpu'

color_samples = [np.array([np.random.randint(0, 255) for _ in range(3)]) for _ in range(100)]


def show_heatmap(heatmap, mask_threshold=None, title=None, save_loc=None):
    plt.figure(figsize=(10,10))
    if title is not None:
        plt.title(title)
    if mask_threshold is not None:
        heatmap = heatmap.copy()
        heatmap[heatmap < mask_threshold] = 0
    plt.imshow(heatmap, cmap='gray', interpolation='nearest', vmin=0, vmax=1)
    plt.axis('off')
    if save_loc is not None:
        plt.savefig(save_loc)    


def save_segment_map(segment_map, classes, save_loc):
    H, W = segment_map.shape
    color_map = np.zeros((H, W, 3), dtype='uint8')
    n_classes = segment_map.max() + 1
    color_masks = []
    for i in range(n_classes):
        if classes[i] in mcolors.cnames.keys():
            h = mcolors.cnames[classes[i]].strip('#')
            color_mask = np.array([int(h[i:i+2], 16) for i in (0, 2, 4)])
        else:
            color_mask = color_samples[i%len(color_samples)]
        color_map[segment_map==i] = color_mask
        color_masks.append((*(color_mask/255.),1))
    handles = [plt.Rectangle((0, 0), 0, 0, color=color_masks[i], label=classes[i]) for i in range(n_classes)]
    plt.legend(handles=handles, title='class')
    plt.imshow(color_map)
    plt.axis('off')
    plt.savefig(save_loc)


def segment_with_threshold(image_shape, similarity, threshold=0.2):
    similarity_argmax = np.stack([similarity[i, ..., i] for i in range(len(similarity))]).argmax(0)
    similarity_max = np.stack([similarity[i, ..., i] for i in range(len(similarity))]).max(0)
    similarity_mask = similarity_max > threshold
    segment_map = -np.ones(image_shape, dtype='int')
    segment_map[similarity_mask] = similarity_argmax[similarity_mask]
    return segment_map


def main(args):
    with open(args.config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    output_dir = config['output_dir']

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    dowmsample = config['dowmsample']
    use_openclip = config['use_openclip']
    image_features = torch.load(config['feature_loc'])

    querys = config['query']
    use_prompt_ensemble = config['use_prompt_ensemble']
    threshold = config['simularity_threshold']
    
    image = cv2.imread(config['image_loc'])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (image.shape[1]//dowmsample, image.shape[0]//dowmsample), interpolation=cv2.INTER_AREA)

    clip = OpenCLIP() if use_openclip else OpenAICLIP()

    similarity, text_probs = clip.predict_similarity_objects_with_feature_attention(image_features, querys, use_prompt_ensemble)

    similarity = similarity.cpu().numpy()
    text_probs = text_probs.cpu().numpy()

    segmentmap_argmax_text_probs = np.stack([text_probs[i, ..., i] for i in range(len(text_probs))]).argmax(0)
    segmentmap_argmax_similarity = np.stack([similarity[i, ..., i] for i in range(len(similarity))]).argmax(0)
    save_segment_map(segmentmap_argmax_text_probs, querys, os.path.join(output_dir, 'argmax_text_probs.png'))
    save_segment_map(segmentmap_argmax_similarity, querys, os.path.join(output_dir, 'argmax_similarity.png'))

    if threshold > 0:
        save_segment_map(segment_with_threshold(image.shape[:2], similarity, threshold), querys, os.path.join(output_dir, f'argmax_similarity_threshold{threshold}.png'))

    with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Per-Pixel Semantic Feature Generator')
    parser.add_argument('--config_path', type=str, required=True, help='path of the config.')

    args = parser.parse_args()

    main(args)