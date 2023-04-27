import argparse
import torch
import cv2
import os

from seganyclip import AutoSegmentAnything, OpenAICLIP, OpenCLIP, AutoSegAnyCLIP

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main(args):
    clip = OpenCLIP() if args.openclip else OpenAICLIP()
    segany = AutoSegmentAnything(args.checkpoint_dir, "default", device)
    segclip = AutoSegAnyCLIP(segany, clip, device)

    image = cv2.imread(args.image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (image.shape[1]//args.downsample, image.shape[0]//args.downsample), interpolation=cv2.INTER_AREA)

    segany.set_image(image)

    output_dir = os.path.join(args.output_dir, 'clip' if not args.openclip else 'openclip')

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    image_features_multires_crop = segclip.encode_image(bbox_crop=args.bbox_crop, extent_segmentation_mask=1)
    torch.save(image_features_multires_crop, 
               os.path.join(output_dir, 
                            f'{args.output_name}_{args.downsample}.pt'))

    del image_features_multires_crop
    torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Per-Pixel Semantic Feature Generator')
    parser.add_argument('--image_path', type=str, required=True, help='path of the image to generate features.')
    parser.add_argument('--output_dir', type=str, required=True, help='path to the directory to save features.')
    parser.add_argument('--output_name', type=str, required=True, help="file name of image features")
    parser.add_argument('--downsample', default=8, type=int, help='the scale of downsampling on the input image.')
    parser.add_argument('--openclip', default=False, type=bool, help='using OpenCLIP as feature extractor instead of openai-CLIP.')
    parser.add_argument('--bbox_crop', default=True, type=bool, help='using bbox instead of segmentation mask to crop the mask before encoding.')
    parser.add_argument('--checkpoint_dir', default='', type=str, help='path to the directory of SAM checkpoints.')
    parser.add_argument('--model_type', default='default', type=str, help='the type of the SAM.')

    args = parser.parse_args()

    main(args)