"""model for MolDiffusion"""

import argparse
from typing import List

import cv2
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from .models import transformers

from .dataset import get_transforms
from .components import Encoder, Decoder
from .chemical import convert_graph_to_smiles
from .tokenization import get_tokenizer

def loading(module, module_states):
    """
    Loads the model's state_dict into a module, handling potential prefix mismatches.
    """
    def remove_prefix(state_dict):
        return {k.replace('module.', ''): v for k, v in state_dict.items()}
    missing_keys, unexpected_keys = module.load_state_dict(remove_prefix(module_states), strict=False)
    return

BOND_TYPES = ["", "single", "double", "triple", "aromatic", "solid wedge", "dashed wedge"]

class moldiffusion:
    """
    Main Interface for MolDiffusion to get predictions
    Args:
        model_path (str): Path to the saved model file.
        device (torch.device): Device to run the model on, defaults to CPU if None.
    """
    def __init__(self, model_path, device=None):
        print(f"Loading model from {model_path}...")
        model_states = torch.load(model_path, map_location=torch.device('cpu'))
        
        # Load args and ensure diffusion params exist
        args = self._get_args(model_states.get('args', {}))
        self.args = args # Save args for inference params

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        
        self.tokenizer = get_tokenizer(args)
        self.encoder, self.decoder = self._get_model(args, self.tokenizer, self.device, model_states)
        
        self.transform = get_transforms(args.input_size, args.input_size, augment=False)
        print("Model loaded successfully.")

    def _get_args(self, args_states=None):
        parser = argparse.ArgumentParser()
        # Model
        parser.add_argument('--encoder', type=str, default='swin_base')
        parser.add_argument('--decoder', type=str, default='diffusion') # Default to diffusion
        parser.add_argument('--no_pretrained', action='store_true')
        parser.add_argument('--use_checkpoint', action='store_true', default=True)
        parser.add_argument('--encoder_dim', type=int, default=0) # will be set by encoder
        
        # Transformer / Diffusion Options
        group = parser.add_argument_group("diffusion_options")
        group.add_argument("--dec_num_layers", type=int, default=12) 
        group.add_argument("--dec_hidden_size", type=int, default=512)
        group.add_argument("--dec_attn_heads", type=int, default=16)
        group.add_argument("--dec_num_queries", type=int, default=128)
        group.add_argument("--hidden_dropout", type=float, default=0.1)
        group.add_argument("--attn_dropout", type=float, default=0.1)
        
        # Diffusion specific params (Important for inference)
        group.add_argument('--decode_steps', type=int, default=480) 
        group.add_argument('--cfg_dropout_prob', type=float, default=0.0)
        group.add_argument('--cfg_guidance_scale', type=float, default=1.0)
        group.add_argument('--temperature', type=float, default=0.0) # Greedy decoding
        group.add_argument('--block_length', type=int, default=4)

        parser.add_argument('--continuous_coords', action='store_true')
        # Data
        parser.add_argument('--input_size', type=int, default=512) # Will be overwritten by saved args
        parser.add_argument('--vocab_file', type=str, default=None)
        parser.add_argument('--coord_bins', type=int, default=64)
        parser.add_argument('--sep_xy', action='store_true', default=True)
        parser.add_argument('--formats', type=str, default="chartok_coords,edges") 

        args = parser.parse_args([])
        
        # Override with saved states
        if args_states:
            for key, value in args_states.items():
                if isinstance(value, list) and key == 'formats':
                    continue 
                args.__dict__[key] = value
                
        if isinstance(args.formats, str):
            args.formats = args.formats.split(',')
            
        return args

    def _get_model(self, args, tokenizer, device, states):
        encoder = Encoder(args, pretrained=False)
        args.encoder_dim = encoder.n_features
        decoder = Decoder(args, tokenizer)

        loading(encoder, states['encoder'])
        loading(decoder, states['decoder'])

        encoder.to(device)
        decoder.to(device)
        encoder.eval()
        decoder.eval()
        return encoder, decoder

    def predict_images(self, input_images: List, return_atoms_bonds=False, return_confidence=False, batch_size=16):
        device = self.device
        predictions = []
        
        # Determine sequence format key (chartok_coords or atomtok_coords)
        seq_format = next((f for f in self.args.formats if f in ['chartok_coords', 'atomtok_coords']), 'chartok_coords')

        for idx in range(0, len(input_images), batch_size):
            batch_images = input_images[idx:idx+batch_size]
            # Preprocess images
            processed_imgs = []
            for image in batch_images:
                if torch.is_tensor(image):
                    image = image.cpu().numpy()
                aug = self.transform(image=image, keypoints=[])
                processed_imgs.append(aug['image'])
            
            images_tensor = torch.stack(processed_imgs, dim=0).to(device)

            with torch.no_grad():
                features, hiddens = self.encoder(images_tensor)
                
                # Call Discrete Diffusion Decoder
                batch_predictions = self.decoder.decode(
                    features, 
                    hiddens, 
                    decode_steps=self.args.decode_steps,
                    temperature=self.args.temperature,
                    guidance_scale=self.args.cfg_guidance_scale,
                    block_length=self.args.block_length
                )
            predictions += batch_predictions

        # Extract results based on the output structure of Decoder.decode
        node_coords = []
        node_symbols = []
        edges = []
        
        for pred in predictions:
            if seq_format in pred:
                node_coords.append(pred[seq_format]['coords'])
                node_symbols.append(pred[seq_format]['symbols'])
            else:
                node_coords.append([])
                node_symbols.append([])
            
            if 'edges' in pred:
                edges.append(pred['edges'])
            else:
                edges.append([])

        # Convert graph to SMILES
        smiles_list, molblock_list, r_success = convert_graph_to_smiles(
            node_coords, node_symbols, edges, images=input_images)

        outputs = []
        for i, (smiles, molfile, pred) in enumerate(zip(smiles_list, molblock_list, predictions)):
            pred_dict = {"predicted_smiles": smiles, "predicted_molfile": molfile}
            
            if return_atoms_bonds and seq_format in pred:
                coords = pred[seq_format]['coords']
                symbols = pred[seq_format]['symbols']
                
                # get atoms info
                atom_list = []

                atom_scores = pred[seq_format].get('atom_scores', [0.0] * len(symbols))
                
                for k, (symbol, coord) in enumerate(zip(symbols, coords)):
                    atom_dict = {
                        "atom_number": f"{k}", 
                        "atom_symbol": symbol, 
                        "coords": (round(coord[0], 3), round(coord[1], 3))
                    }
                    if return_confidence:
                        atom_dict["confidence"] = atom_scores[k]
                    atom_list.append(atom_dict)
                pred_dict["atom_sets"] = atom_list
                
                # get bonds info
                if 'edges' in pred:
                    bond_list = []
                    num_atoms = len(symbols)
                    edge_preds = pred['edges'] 
                    
                    for row in range(min(len(edge_preds), num_atoms)):
                        for col in range(row + 1, min(len(edge_preds), num_atoms)):
                            bond_type_int = edge_preds[row][col]
                            if torch.is_tensor(bond_type_int):
                                bond_type_int = bond_type_int.item()
                                
                            if bond_type_int != 0 and bond_type_int < len(BOND_TYPES):
                                bond_type_str = BOND_TYPES[bond_type_int]
                                bond_dict = {
                                    "atom_number": f"{row}",
                                    "bond_type": bond_type_str, 
                                    "endpoints": (row, col)
                                }
                                bond_list.append(bond_dict)
                    pred_dict["bond_sets"] = bond_list
                    
            outputs.append(pred_dict)
        return outputs

    def predict_image(self, image, return_atoms_bonds=False, return_confidence=False):
        return self.predict_images([image], return_atoms_bonds=return_atoms_bonds, return_confidence=return_confidence)[0]

    def predict_image_files(self, image_files: List, return_atoms_bonds=False, return_confidence=False):
        input_images = []
        for path in image_files:
            image = cv2.imread(path)
            if image is None:
                print(f"Warning: Could not read image at {path}")
                # Create a dummy white image
                image = np.full((512, 512, 3), 255, dtype=np.uint8)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            input_images.append(image)
        return self.predict_images(input_images, return_atoms_bonds=return_atoms_bonds, return_confidence=return_confidence)

    def predict_final_results(self, image_file: str, return_atoms_bonds=False, return_confidence=False):
        return self.predict_image_files([image_file], return_atoms_bonds=return_atoms_bonds, return_confidence=return_confidence)[0]