import os
import torch
from .tokenization import PAD_ID, EOS_ID, MASK_ID

class VisualizationLogger:
    def __init__(self, save_path, tokenizer, rank=0):
        self.save_path = save_path
        self.tokenizer = tokenizer
        self.rank = rank
        self.log_dir = os.path.join(save_path, 'visualization_logs')
        if self.rank == 0:
            os.makedirs(self.log_dir, exist_ok=True)

    def _sequence_to_string(self, seq_tensor):
        if seq_tensor.dim() == 0:
            seq_tensor = seq_tensor.unsqueeze(0)
            
        token_list = []
        for token_id in seq_tensor.cpu().numpy():
            token_str = ""
            if self.tokenizer.is_x(token_id):
                coord_val = self.tokenizer.id_to_x(token_id)
                token_str = f'X({coord_val:.2f})'
            elif self.tokenizer.is_y(token_id):
                coord_val = self.tokenizer.id_to_y(token_id)
                token_str = f'Y({coord_val:.2f})'
            else:
                token_str = self.tokenizer.itos.get(token_id, f'[{token_id}]')
            token_list.append(token_str)
            
            # if token_id in [PAD_ID, EOS_ID]:
            #     break
        
        return " ".join(token_list)

    def log_train_step(self, epoch, step, x0, xt, pred_x0_logits, global_step):
        if self.rank != 0 or global_step % 500 != 0:  
            return

        x0_sample = x0[0]
        xt_sample = xt[0]
        pred_x0_sample = torch.argmax(pred_x0_logits[0], dim=-1)

        log_content = (
            f"--- Training Visualization (Epoch: {epoch+1}, Step: {step}, GlobalStep: {global_step}) ---\n"
            f"Ground Truth (x0): {self._sequence_to_string(x0_sample)}\n"
            f"Noised Input (xt): {self._sequence_to_string(xt_sample)}\n"
            f"Model Pred (x0^): {self._sequence_to_string(pred_x0_sample)}\n"
            f"{'='*80}\n\n"
        )
        
        log_file = os.path.join(self.log_dir, 'train_process.log')
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(log_content)

    def log_inference_start(self, initial_xt, inference_step_count):
        if self.rank != 0:
            return

        log_content = (
            f"\n\n{'#'*30} New Inference Run {'#'*30}\n"
            f"Total Inference Steps: {inference_step_count}\n"
            f"Initial Noise (x_T): {self._sequence_to_string(initial_xt[0])}\n"
            f"{'-'*80}\n"
        )
        
        log_file = os.path.join(self.log_dir, 'inference_process.log')
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(log_content)

    def log_inference_step(self, t, xt):
        if self.rank != 0:
            return

        log_content = (
            f"Step t={t}:\n"
            f"  Output (x_{t}): {self._sequence_to_string(xt[0])}\n"
        )
        
        log_file = os.path.join(self.log_dir, 'inference_process.log')
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(log_content)