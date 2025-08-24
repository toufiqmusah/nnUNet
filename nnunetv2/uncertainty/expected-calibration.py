import os
import warnings
import numpy as np
import seaborn as sns
import nibabel as nib
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from typing import Dict, List, Tuple, Optional

warnings.filterwarnings('ignore')

class ECECalculator:
    """
    Expected Calibration Error calculator for binary segmentation with uncertainty quantification
    """
    
    def __init__(self, n_bins: int = 15):
        self.n_bins = n_bins
        self.bin_boundaries = np.linspace(0, 1, n_bins + 1)
        self.bin_lowers = self.bin_boundaries[:-1]
        self.bin_uppers = self.bin_boundaries[1:]
    
    def load_nifti_predictions(self, pred_dir: str, file_pattern: str = "*.png") -> Dict[str, np.ndarray]:
        """Load prediction files from directory"""
        pred_files = {}
        pred_path = Path(pred_dir)
        
        if not pred_path.exists():
            raise FileNotFoundError(f"Prediction directory not found: {pred_dir}")
        
        # finding all prediction files
        prediction_files = list(pred_path.glob(file_pattern))
        if not prediction_files:
            prediction_files = list(pred_path.glob("*.png"))
        
        print(f"Found {len(prediction_files)} files in {pred_dir}")
        
        for file_path in prediction_files:
            try:
                # Load file
                nii_img = nib.load(str(file_path))
                data = nii_img.get_fdata()
                
                # Handle different data shapes (ensure we get probabilities)
                if data.ndim == 3 and data.shape[-1] == 2:  # Multi-class probability format
                    # Take foreground class probability
                    prob_data = data[..., 1]
                elif data.ndim == 2:  # Single probability map
                    prob_data = data
                else:
                    print(f"Warning: Unexpected data shape {data.shape} for {file_path.name}")
                    continue
                
                # Store with case identifier
                case_id = file_path.stem.replace('.nii', '').replace('_probabilities', '')
                pred_files[case_id] = prob_data
                
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
        
        return pred_files
    
    def aggregate_monte_carlo_predictions(self, base_dir: str = "Monte-Carlo-Predictions") -> Dict[str, np.ndarray]:
        """Aggregate Monte Carlo dropout predictions by averaging"""
        print("Loading Monte Carlo Dropout predictions...")
        
        all_predictions = {}
        
        # Get all prediction directories
        pred_dirs = [d for d in os.listdir(base_dir) if d.startswith('Prediction-')]
        pred_dirs.sort()
        
        print(f"Found {len(pred_dirs)} Monte Carlo prediction sets")
        
        for pred_dir in pred_dirs:
            pred_path = os.path.join(base_dir, pred_dir)
            predictions = self.load_nifti_predictions(pred_path)
            
            for case_id, prob_map in predictions.items():
                if case_id not in all_predictions:
                    all_predictions[case_id] = []
                all_predictions[case_id].append(prob_map)
        
        # Average predictions across Monte Carlo samples
        aggregated_predictions = {}
        for case_id, prob_maps in all_predictions.items():
            if len(prob_maps) > 0:
                aggregated_predictions[case_id] = np.mean(prob_maps, axis=0)
                print(f"Aggregated {len(prob_maps)} MC samples for {case_id}")
        
        return aggregated_predictions
    
    def aggregate_deep_ensemble_predictions(self, base_dir: str = "Deep-Ensemble-Predictions") -> Dict[str, np.ndarray]:
        """Aggregate Deep Ensemble predictions by averaging across folds"""
        print("Loading Deep Ensemble predictions...")
        
        all_predictions = {}
        
        # Get all prediction directories
        pred_dirs = [d for d in os.listdir(base_dir) if d.startswith('Prediction-')]
        pred_dirs.sort()
        
        print(f"Found {len(pred_dirs)} Deep Ensemble prediction sets")
        
        for pred_dir in pred_dirs:
            pred_path = os.path.join(base_dir, pred_dir)
            predictions = self.load_nifti_predictions(pred_path)
            
            for case_id, prob_map in predictions.items():
                if case_id not in all_predictions:
                    all_predictions[case_id] = []
                all_predictions[case_id].append(prob_map)
        
        # Average predictions across ensemble members
        aggregated_predictions = {}
        for case_id, prob_maps in all_predictions.items():
            if len(prob_maps) > 0:
                aggregated_predictions[case_id] = np.mean(prob_maps, axis=0)
                print(f"Aggregated {len(prob_maps)} ensemble members for {case_id}")
        
        return aggregated_predictions
    
    def aggregate_deep_ensemble_mc_predictions(self, base_dir: str = "Deep-Ensemble-Monte-Carlo-Predictions") -> Dict[str, np.ndarray]:
        """Aggregate Deep Ensemble Monte Carlo predictions"""
        print("Loading Deep Ensemble Monte Carlo predictions...")
        
        all_predictions = {}
        
        # Get all prediction directories
        pred_dirs = [d for d in os.listdir(base_dir) if d.startswith('Prediction-F')]
        pred_dirs.sort()
        
        print(f"Found {len(pred_dirs)} Deep Ensemble Monte Carlo prediction sets")
        
        for pred_dir in pred_dirs:
            pred_path = os.path.join(base_dir, pred_dir)
            predictions = self.load_nifti_predictions(pred_path)
            
            for case_id, prob_map in predictions.items():
                if case_id not in all_predictions:
                    all_predictions[case_id] = []
                all_predictions[case_id].append(prob_map)
        
        # Average predictions across all ensemble members and MC samples
        aggregated_predictions = {}
        for case_id, prob_maps in all_predictions.items():
            if len(prob_maps) > 0:
                aggregated_predictions[case_id] = np.mean(prob_maps, axis=0)
                print(f"Aggregated {len(prob_maps)} ensemble MC samples for {case_id}")
        
        return aggregated_predictions
    
    def load_ground_truth(self, gt_dir: str) -> Dict[str, np.ndarray]:
        """Load ground truth segmentation masks"""
        print("Loading ground truth masks...")
        
        gt_files = {}
        gt_path = Path(gt_dir)
        
        if not gt_path.exists():
            raise FileNotFoundError(f"Ground truth directory not found: {gt_dir}")
        
        # Find all nifti files
        prediction_files = list(gt_path.glob("*.nii.gz"))
        if not prediction_files:
            prediction_files = list(gt_path.glob("*.nii"))
        
        print(f"Found {len(prediction_files)} ground truth files")
        
        for file_path in prediction_files:
            try:
                nii_img = nib.load(str(file_path))
                data = nii_img.get_fdata()
                
                # Ensure binary mask (0 and 1)
                if data.max() > 1:
                    data = (data > 0).astype(np.float32)
                
                case_id = file_path.stem.replace('.nii', '')
                gt_files[case_id] = data.astype(np.uint8)
                
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
        
        return gt_files
    
    def calculate_ece(self, predictions: Dict[str, np.ndarray], 
                     ground_truth: Dict[str, np.ndarray]) -> Tuple[float, Dict]:
        """Calculate Expected Calibration Error"""
        
        all_confidences = []
        all_accuracies = []
        
        # Match predictions with ground truth
        matched_cases = set(predictions.keys()) & set(ground_truth.keys())
        print(f"Matched {len(matched_cases)} cases for ECE calculation")
        
        if len(matched_cases) == 0:
            raise ValueError("No matching cases found between predictions and ground truth")
        
        for case_id in matched_cases:
            pred_probs = predictions[case_id]
            gt_mask = ground_truth[case_id]
            
            # Ensure same shape
            if pred_probs.shape != gt_mask.shape:
                print(f"Shape mismatch for {case_id}: pred {pred_probs.shape}, gt {gt_mask.shape}")
                continue
            
            # Flatten arrays
            pred_flat = pred_probs.flatten()
            gt_flat = gt_mask.flatten()
            
            # Get confidence (max probability) and predicted class
            confidences = np.maximum(pred_flat, 1 - pred_flat)  # Confidence in predicted class
            predicted_class = (pred_flat > 0.5).astype(int)
            
            # Calculate pixel-wise accuracy
            pixel_accuracies = (predicted_class == gt_flat).astype(int)
            
            all_confidences.extend(confidences)
            all_accuracies.extend(pixel_accuracies)
        
        all_confidences = np.array(all_confidences)
        all_accuracies = np.array(all_accuracies)
        
        # Calculate ECE using binning
        ece = 0
        bin_details = []
        
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Find predictions in this confidence bin
            in_bin = (all_confidences > bin_lower) & (all_confidences <= bin_upper)
            prop_in_bin = in_bin.sum() / len(all_confidences)
            
            if prop_in_bin > 0:
                # Calculate accuracy and average confidence for this bin
                accuracy_in_bin = all_accuracies[in_bin].mean()
                avg_confidence_in_bin = all_confidences[in_bin].mean()
                
                # Add to ECE
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                
                bin_details.append({
                    'bin_lower': bin_lower,
                    'bin_upper': bin_upper,
                    'accuracy': accuracy_in_bin,
                    'confidence': avg_confidence_in_bin,
                    'proportion': prop_in_bin,
                    'count': in_bin.sum()
                })
            else:
                bin_details.append({
                    'bin_lower': bin_lower,
                    'bin_upper': bin_upper,
                    'accuracy': 0,
                    'confidence': 0,
                    'proportion': 0,
                    'count': 0
                })
        
        return ece, {
            'bin_details': bin_details,
            'total_pixels': len(all_confidences),
            'overall_accuracy': all_accuracies.mean()
        }
    
    def plot_calibration_curve(self, results: Dict, method_name: str, save_path: Optional[str] = None):
        """Plot calibration curve"""
        
        plt.figure(figsize=(10, 8))
        
        # Extract data for plotting
        bin_centers = []
        accuracies = []
        confidences = []
        counts = []
        
        for bin_info in results['bin_details']:
            if bin_info['count'] > 0:
                bin_center = (bin_info['bin_lower'] + bin_info['bin_upper']) / 2
                bin_centers.append(bin_center)
                accuracies.append(bin_info['accuracy'])
                confidences.append(bin_info['confidence'])
                counts.append(bin_info['count'])
        
        # Plot calibration curve
        plt.subplot(2, 2, 1)
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect calibration')
        plt.scatter(confidences, accuracies, s=np.array(counts)/1000, alpha=0.7, label='Bin accuracy')
        plt.xlabel('Confidence')
        plt.ylabel('Accuracy')
        plt.title(f'Calibration Curve - {method_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot confidence histogram
        plt.subplot(2, 2, 2)
        plt.bar(bin_centers, counts, width=0.05, alpha=0.7, edgecolor='black')
        plt.xlabel('Confidence')
        plt.ylabel('Number of Predictions')
        plt.title('Confidence Distribution')
        plt.grid(True, alpha=0.3)
        
        # Plot accuracy per bin
        plt.subplot(2, 2, 3)
        plt.bar(bin_centers, accuracies, width=0.05, alpha=0.7, edgecolor='black')
        plt.xlabel('Confidence Bin')
        plt.ylabel('Accuracy')
        plt.title('Accuracy per Confidence Bin')
        plt.grid(True, alpha=0.3)
        
        # Plot ECE contribution per bin
        plt.subplot(2, 2, 4)
        ece_contributions = [abs(conf - acc) * count/sum(counts) 
                           for conf, acc, count in zip(confidences, accuracies, counts)]
        plt.bar(bin_centers, ece_contributions, width=0.05, alpha=0.7, edgecolor='black')
        plt.xlabel('Confidence Bin')
        plt.ylabel('ECE Contribution')
        plt.title('ECE Contribution per Bin')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Calibration plot saved to: {save_path}")
        
        plt.show()

def main():
    """Main function to calculate ECE for all uncertainty methods"""
    
    # Initialize ECE calculator
    ece_calc = ECECalculator(n_bins=15)
    
    # Define paths - modify these according to your setup
    gt_dir = "ground_truth"  # Directory containing ground truth masks
    
    # Dictionary to store results
    results = {}
    
    try:
        # Load ground truth
        print("="*50)
        print("LOADING GROUND TRUTH")
        print("="*50)
        ground_truth = ece_calc.load_ground_truth(gt_dir)
        
        if len(ground_truth) == 0:
            print("No ground truth files found. Please check the ground truth directory path.")
            return
        
        # 1. Monte Carlo Dropout
        print("\n" + "="*50)
        print("PROCESSING MONTE CARLO DROPOUT")
        print("="*50)
        try:
            mc_predictions = ece_calc.aggregate_monte_carlo_predictions()
            if len(mc_predictions) > 0:
                mc_ece, mc_details = ece_calc.calculate_ece(mc_predictions, ground_truth)
                results['Monte Carlo Dropout'] = {
                    'ece': mc_ece,
                    'details': mc_details
                }
                print(f"Monte Carlo Dropout ECE: {mc_ece:.4f}")
                
                # Plot calibration curve
                ece_calc.plot_calibration_curve(mc_details, "Monte Carlo Dropout", 
                                              "monte_carlo_calibration.png")
            else:
                print("No Monte Carlo predictions found")
        except Exception as e:
            print(f"Error processing Monte Carlo predictions: {e}")
        
        # 2. Deep Ensemble
        print("\n" + "="*50)
        print("PROCESSING DEEP ENSEMBLE")
        print("="*50)
        try:
            de_predictions = ece_calc.aggregate_deep_ensemble_predictions()
            if len(de_predictions) > 0:
                de_ece, de_details = ece_calc.calculate_ece(de_predictions, ground_truth)
                results['Deep Ensemble'] = {
                    'ece': de_ece,
                    'details': de_details
                }
                print(f"Deep Ensemble ECE: {de_ece:.4f}")
                
                # Plot calibration curve
                ece_calc.plot_calibration_curve(de_details, "Deep Ensemble", 
                                              "deep_ensemble_calibration.png")
            else:
                print("No Deep Ensemble predictions found")
        except Exception as e:
            print(f"Error processing Deep Ensemble predictions: {e}")
        
        # 3. Deep Ensemble Monte Carlo
        print("\n" + "="*50)
        print("PROCESSING DEEP ENSEMBLE MONTE CARLO")
        print("="*50)
        try:
            demc_predictions = ece_calc.aggregate_deep_ensemble_mc_predictions()
            if len(demc_predictions) > 0:
                demc_ece, demc_details = ece_calc.calculate_ece(demc_predictions, ground_truth)
                results['Deep Ensemble Monte Carlo'] = {
                    'ece': demc_ece,
                    'details': demc_details
                }
                print(f"Deep Ensemble Monte Carlo ECE: {demc_ece:.4f}")
                
                # Plot calibration curve
                ece_calc.plot_calibration_curve(demc_details, "Deep Ensemble Monte Carlo", 
                                              "deep_ensemble_mc_calibration.png")
            else:
                print("No Deep Ensemble Monte Carlo predictions found")
        except Exception as e:
            print(f"Error processing Deep Ensemble Monte Carlo predictions: {e}")
        
        # Summary
        print("\n" + "="*50)
        print("RESULTS SUMMARY")
        print("="*50)
        for method, result in results.items():
            print(f"{method}: ECE = {result['ece']:.4f}")
            print(f"  Overall Accuracy: {result['details']['overall_accuracy']:.4f}")
            print(f"  Total Pixels: {result['details']['total_pixels']:,}")
            print()
        
        # Create comparison plot
        if len(results) > 1:
            plt.figure(figsize=(10, 6))
            methods = list(results.keys())
            eces = [results[method]['ece'] for method in methods]
            accuracies = [results[method]['details']['overall_accuracy'] for method in methods]
            
            x = np.arange(len(methods))
            width = 0.35
            
            plt.subplot(1, 2, 1)
            plt.bar(x, eces, width, alpha=0.8)
            plt.xlabel('Method')
            plt.ylabel('Expected Calibration Error')
            plt.title('ECE Comparison')
            plt.xticks(x, methods, rotation=45)
            plt.grid(True, alpha=0.3)
            
            plt.subplot(1, 2, 2)
            plt.bar(x, accuracies, width, alpha=0.8, color='green')
            plt.xlabel('Method')
            plt.ylabel('Overall Accuracy')
            plt.title('Accuracy Comparison')
            plt.xticks(x, methods, rotation=45)
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('ece_comparison.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print("Comparison plot saved as 'ece_comparison.png'")
    
    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()