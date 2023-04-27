import argparse
import torch
import cv2
import os
import yaml
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from seganyclip import OpenAICLIP, OpenCLIP, AutoSegmentAnything, AutoSegAnyCLIP

device = 'cuda' if torch.cuda.is_available() else 'cpu'

np.random.seed(2)
color_samples = [np.array([np.random.randint(0, 255) for _ in range(3)]) for _ in range(100)]

def save_segment_map(segment_map, classes, save_loc):
    H, W = segment_map.shape
    color_map = np.ones((H, W, 3), dtype='uint8') * 155
    n_classes = len(classes)
    classes = classes.copy()
    color_masks = []
    for i in range(n_classes):
        if classes[i] in mcolors.cnames.keys():
            h = mcolors.cnames[classes[i]].strip('#')
            color_mask = np.array([int(h[i:i+2], 16) for i in (0, 2, 4)])
        else:
            color_mask = color_samples[i%len(color_samples)]
        color_map[segment_map==i] = color_mask
        color_masks.append((*(color_mask/255.),1))
    classes.insert(0, 'none')
    color_masks.insert(0, (0.6, 0.6, 0.6, 1))
    handles = [plt.Rectangle((0, 0), 0, 0, color=color_masks[i], label=classes[i]) for i in range(n_classes+1)]
    plt.legend(handles=handles, framealpha=0.4, title='class')
    plt.imshow(color_map)
    plt.axis('off')
    plt.savefig(save_loc, bbox_inches='tight', pad_inches=0)
    

def main(args):
    with open(args.config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    output_dir = config['output_dir']

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    feature_loc = config['feature_loc']

    use_openclip = config['use_openclip']

    querys = config['query']
    use_prompt_ensemble = config['use_prompt_ensemble']
    batch_size = config['batch_size']
    top_k = config['top_k']
    threshold = config['similarity_threshold']

    assert(top_k >= 1)

    if os.path.exists(feature_loc) and not config['replace_feature']:
        print(f'Load model.')
        clip = OpenCLIP() if use_openclip else OpenAICLIP()
        print(f'Load features from {feature_loc}.')
        image_features = torch.load(feature_loc)
    else:
        print(f'Load model.')
        clip = OpenCLIP() if use_openclip else OpenAICLIP()
        segany = AutoSegmentAnything(config['sam_checkpoint_loc'], "default", device)
        segclip = AutoSegAnyCLIP(segany, clip, device)

        print(f'Not find features at {feature_loc}.')
        print(f'Generate and store features at \'{feature_loc}\'.')

        image_loc = config['image_loc']
        dowmsample = config['dowmsample']
        resolution = config['resolution']

        assert(resolution > 0 and resolution <= 3)
    
        image = cv2.imread(image_loc)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (image.shape[1]//dowmsample, image.shape[0]//dowmsample), interpolation=cv2.INTER_AREA)

        segany.set_image(image)
        segclip.n_objs = resolution
        image_features = segclip.encode_image(extent_segmentation_mask=1, bbox_crop=False)

        torch.save(image_features, feature_loc)

        del segany, segclip
        torch.cuda.empty_cache()

    similarity_argmax_top_k = clip.predict_similarity_objects_with_feature_attention_batch(image_features, querys, use_prompt_ensemble, batch_size, top_k, threshold)

    similarity_argmax_top_k  = [similarity_argmax.cpu().numpy() for similarity_argmax in similarity_argmax_top_k]

    for k, similarity_argmax in enumerate(similarity_argmax_top_k):
        save_segment_map(similarity_argmax, querys, os.path.join(output_dir, f'similarity_max{k}.png'))

    with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Per-Pixel Semantic Feature Generator')
    parser.add_argument('--config_path', type=str, required=True, help='path of the config.')

    args = parser.parse_args()

    main(args)