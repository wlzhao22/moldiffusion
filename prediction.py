import argparse
import json
import torch
import os
from moldiffusion.model import moldiffusion

import warnings 
warnings.filterwarnings('ignore')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Updated default path to your best checkpoint
    parser.add_argument('--model_path', type=str, required=True, help="Path to the .pth model checkpoint")
    parser.add_argument('--image_path', type=str, required=True, help="Path to the input image file")
    
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        exit(1)
        
    if not os.path.exists(args.image_path):
        print(f"Error: Image file not found at {args.image_path}")
        exit(1)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        model = moldiffusion(args.model_path, device)
        
        print(f"Predicting for image: {args.image_path}")
        output = model.predict_final_results(
            args.image_path, 
            return_atoms_bonds=True
        )
        
        print("-" * 30)
        print("Prediction Results:")
        print("-" * 30)
        for key, value in output.items():
            if key in ["atom_sets", "bond_sets"]:
                print(f"{key}: [List containing {len(value)} elements]")
            else:
                print(f"{key}:")
                print(value + '\n' if isinstance(value, str) else json.dumps(value) + '\n')
                
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Prediction failed: {e}")