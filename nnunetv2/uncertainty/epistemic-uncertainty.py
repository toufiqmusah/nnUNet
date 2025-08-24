# How Should this work? 
# Starting with Monte-Carlo Dropout Inference
# After user trains unet with dropout, with all/necessary folds saved in nnUNet_Results

# Call [nnUncertainty [function], monte-carlo[uncertainty-type], -f[fold to use], -n[number of stochastic forward passes], -model[model_directory], -o[output directory]]
# Yeaahh, no. One type of uncertainty at a time, per file.


def uncertainty_args():
    import argparse
    parser = argparse.ArgumentParser(description= "Monte-Carlo Dropout Uncertainty Inference for nnUNet")
    parser.add_argument('-f', type=int, default=0, help='Fold to use for inference (default: 0)')
    parser.add_argument('-n', type=int, default=10, help="Number of Stochastic Forward Passes (default: 10)")
    parser.add_argument('-m', type=str, required=True,
                        help='Folder in which the trained model is. Must have subfolders fold_X for the different '
                             'folds you trained')
    parser.add_argument('-i', type=str, required=True,
                        help='input folder. Remember to use the correct channel numberings for your files (_0000 etc). '
                             'File endings must be the same as the training dataset!')


import os
import torch
import numpy as np




from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

def make_nnUNet_inference_list(data_dir: str):
    nnunet_images = []
    for root, dirs, items in os.walk(data_dir):
        for item in items:
            if item.endswith('.png'):
              img_path = os.path.join(root, item)
              nnunet_images.append([img_path])
    return nnunet_images


data_dir = 'BraTS2024-SSA-Challenge-ValidationData'
nnunet_images = make_nnUNet_inference_list(data_dir)
print("nnUNet input image list:", nnunet_images[2])

output_dir_nnunet = "labelsTs"
os.makedirs(output_dir_nnunet, exists_ok=True)

# performing monte-carlo dropout inference on best model for 10 sets of predictions

def monte_carlo_inference(model_folder, dataset_dir, set_of_predictions=10):
  os.makedirs("Monte-Carlo-Predictions", exist_ok=True)

  predictor = nnUNetPredictor(
    tile_step_size=0.5,
    use_gaussian=True,
    use_mirroring=True,
    perform_everything_on_device=True,
    device=torch.device('cuda'),
    verbose=False,
    verbose_preprocessing=False,
    allow_tqdm=True
  )

  predictor.initialize_from_trained_model_folder(
  model_folder,
  use_folds=(0,),
  checkpoint_name='checkpoint_best.pth')

  for n in range(set_of_predictions):
    output_dir_nnunet = f"Monte-Carlo-Predictions/Prediction-{n}"
    os.makedirs(output_dir_nnunet, exist_ok=True)

    print(f"Prection Set: {n}")
    ret = predictor.predict_from_files_sequential(
    make_nnUNet_inference_list(dataset_dir),
    output_dir_nnunet,
    overwrite=True,
    save_probabilities=True,
    folder_with_segs_from_prev_stage=None
    )

    print("Predictions saved to:", os.listdir(output_dir_nnunet))

model_folder = 'nnUNetTrainer_750e'
dataset_dir = 'imagesTs'

monte_carlo_inference(model_folder, dataset_dir)

