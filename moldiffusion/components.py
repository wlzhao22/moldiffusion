# components_discrete_diff.py

import math
import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torch.nn.utils.rnn import pad_sequence 

from .utils import FORMAT_INFO, to_device, sample_tokens
from .tokenization import SOS, EOS, PAD, MASK, SOS_ID, EOS_ID, PAD_ID, MASK_ID 

# ENCODER 
class Encoder(nn.Module):
    def __init__(self, args, pretrained=False):
        super().__init__()
        model_name = args.encoder
        self.model_name = model_name
        if model_name.startswith('resnet'):
            self.model_type = 'resnet'
            self.cnn = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
            self.n_features = self.cnn.num_features  # encoder_dim
            self.cnn.global_pool = nn.Identity()
            self.cnn.fc = nn.Identity()
        elif model_name.startswith('convnext'):
            self.model_type = 'convnext'
            self.cnn = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
            self.n_features = 1024  # encoder_dim
            # self.cnn.global_pool = nn.Identity()
            self.cnn.fc = nn.Identity()
        elif model_name.startswith('swin'):
            self.model_type = 'vision_transformer'
            self.transformer = timm.create_model(model_name, pretrained=pretrained, pretrained_strict=False,
                                                 use_checkpoint=args.use_checkpoint, num_classes=0)
            self.n_features = self.transformer.num_features
            self.transformer.head = nn.Identity()
        else:
            raise NotImplemented

    def forwards(self, transformer, x):
        x, H, W = transformer.patch_embed(x)
        
        if transformer.absolute_pos_embed is not None:
            x = x + transformer.absolute_pos_embed
        x = transformer.pos_drop(x)
        
        hiddens = []
        
        for layer in transformer.layers:
            x, H, W = layer(x, H, W, hiddens)
            
        x = transformer.norm(x)
        
        hiddens[-1] = x
        return x, hiddens

    def forward(self, x, refs=None):
        if self.model_type in ['resnet', 'efficientnet', 'convnext']:
            features = self.cnn(x)
            features = features.flatten(2).transpose(1, 2)
            hiddens = []
        elif self.model_type == 'vision_transformer':
            features, hiddens = self.forwards(self.transformer, x)
        else:
            raise NotImplemented
        return features, hiddens

    

# DECODER and its Components 
class GaussianFourierProjection(nn.Module):
    """Gaussian Fourier features for encoding continuous values."""
    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
    def forward(self, t):
        t_proj = t.unsqueeze(-1) * self.W.unsqueeze(0) * 2 * math.pi
        return torch.cat([torch.sin(t_proj), torch.cos(t_proj)], dim=-1)


class Embeddings(nn.Module):
    """
    Standard word and positional embeddings for the Transformer decoder.
    """
    def __init__(self, word_vec_size, word_vocab_size, word_padding_idx, position_encoding=True, dropout=0.1):
        super(Embeddings, self).__init__()
        self.word_padding_idx = word_padding_idx
        self.word_embeddings = nn.Embedding(word_vocab_size, word_vec_size, padding_idx=word_padding_idx)
        self.position_encoding = position_encoding
        if self.position_encoding:
            self.position_embeddings = nn.Embedding(2048, word_vec_size) # A large enough max sequence length
        self.layer_norm = nn.LayerNorm(word_vec_size, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, source, step=None):
        if self.position_encoding:
            if step is None:
                positions = torch.arange(source.size(1), device=source.device, dtype=torch.long).expand(source.size(0), -1)
            else:
                positions = torch.full((source.size(0), 1), step, device=source.device, dtype=torch.long)
            pos_emb = self.position_embeddings(positions)
        else:
            pos_emb = 0

        word_emb = self.word_embeddings(source)
        emb = self.layer_norm(word_emb + pos_emb)
        emb = self.dropout(emb)
        return emb
    
class FeatureFusion(nn.Module):
    def __init__(self, stage3_dim=512, stage4_dim=1024):
        super().__init__()
        self.unshuffle = nn.PixelUnshuffle(downscale_factor=2)
        folded_dim = stage3_dim * 4
        
        self.proj = nn.Sequential(
            nn.LayerNorm(folded_dim),
            nn.Linear(folded_dim, stage4_dim, bias=False),
            nn.GELU(),
            nn.LayerNorm(stage4_dim)
        )
        
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, feat_stage3, feat_stage4):
        B, L3, C3 = feat_stage3.shape
        H3 = W3 = int(math.sqrt(L3))
        
        x3 = feat_stage3.transpose(1, 2).view(B, C3, H3, W3)
        
        x3_folded = self.unshuffle(x3)
        x3_seq = x3_folded.flatten(2).transpose(1, 2)
        x3_proj = self.proj(x3_seq)
        
        fused_feat = feat_stage4 + self.gamma * x3_proj
        return fused_feat

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class DiTDecoderBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        # Self-Attention
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn1 = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        
        # Cross-Attention
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn2 = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        
        # MLP
        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, hidden_size),
            nn.Dropout(dropout)
        )
        
        # 3 layers * 3 parameters (shift, scale, gate) = 9
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 9 * hidden_size)
        )
        
        # Zero-Init
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        
    def forward(self, x, c, memory, tgt_key_padding_mask=None):
        """
        x: (B, L, D) - input sequence
        c: (B, D) - Time Embedding
        memory: (B, L_enc, D) - Encoder output
        tgt_key_padding_mask: (B, L) - mask for the target sequence
        """
        shift_msa, scale_msa, gate_msa, \
        shift_ca, scale_ca, gate_ca, \
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(9, dim=1)
        
        # Self-Attention Block
        x_norm = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out = self.attn1(x_norm, x_norm, x_norm, key_padding_mask=tgt_key_padding_mask)[0]
        x = x + gate_msa.unsqueeze(1) * attn_out
        
        # Cross-Attention Block
        x_norm = modulate(self.norm2(x), shift_ca, scale_ca)
        attn_out = self.attn2(x_norm, memory, memory)[0]
        x = x + gate_ca.unsqueeze(1) * attn_out
        
        # MLP Block
        x_norm = modulate(self.norm3(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_norm)
        
        return x


class DiffusionDecoder(nn.Module):
    """
    Non-Autoregressive Transformer Decoder.
    It predicts the entire sequence at once by denoising from a corrupted input.
    """
    def __init__(self, args, tokenizer, format_name, input_dim=None, num_patches=None):
        super().__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.format_name = format_name
        self.vocab_size = len(self.tokenizer)

        if input_dim is None:
            input_dim = args.encoder_dim
        if num_patches is None:
            num_patches = (args.input_size // 32)**2
        self.p_uncond = args.cfg_dropout_prob
        if self.p_uncond > 0:
            self.unconditional_embedding = nn.Parameter(torch.randn(1, num_patches, input_dim))
        else:
            self.unconditional_embedding = None
        
        # Token and position embeddings
        self.embeddings = Embeddings(
            word_vec_size=args.dec_hidden_size,
            word_vocab_size=self.vocab_size,
            word_padding_idx=PAD_ID,
            position_encoding=True,
            dropout=args.hidden_dropout
        )

        # Time step embedding
        self.step_emb = nn.Sequential(
            GaussianFourierProjection(args.dec_hidden_size),
            nn.Linear(args.dec_hidden_size, args.dec_hidden_size),
            nn.SiLU(),
            nn.Linear(args.dec_hidden_size, args.dec_hidden_size)
        )
        
        # Encoder feature projection
        self.enc_trans_layer = nn.Linear(input_dim, args.dec_hidden_size)

        # Core Transformer decoder layers
        self.transformer_layers = nn.ModuleList([
            DiTDecoderBlock(
                hidden_size=args.dec_hidden_size,
                num_heads=args.dec_attn_heads,
                mlp_ratio=4.0,
                dropout=args.hidden_dropout
            ) for _ in range(args.dec_num_layers)
        ])
        
        # Final output layer
        self.output_layer = nn.Linear(args.dec_hidden_size, self.vocab_size)

    def _run_decoder(self, encoder_out, xt, t):
        """ The core denoising model that predicts x0 logits from xt. """
        memory_bank = self.enc_trans_layer(encoder_out)
        tgt_pad_mask = (xt == PAD_ID)
        token_emb = self.embeddings(xt)
        time_emb = self.step_emb(t)
        
        # Combine token embeddings and time embeddings
        x = token_emb
        
        for layer in self.transformer_layers:
            x = layer(
                x=x,
                c = time_emb,
                memory=memory_bank,
                tgt_key_padding_mask=tgt_pad_mask
            )
        dec_out = x

        logits = self.output_layer(dec_out)
        return logits, dec_out
    
    def forward(self, encoder_out, x0):
        """Training with Masked Language Modeling objective and cosine masking schedule."""
        B, L = x0.shape
        device = x0.device
        
        t = torch.rand(B, device=device) # Sample random time t in [0, 1]
        # each token is masked with probability t
        noise = torch.rand(B, L, device=device)
        mask = noise < t.unsqueeze(1) # Expand t for broadcasting
        
        # Protect SOS token from being masked
        mask[:, 0] = False
        
        xt = x0.clone()
        labels = torch.full_like(x0, -100)
        
        xt[mask] = MASK_ID
        labels[mask] = x0[mask]
        
        logits, dec_out = self._run_decoder(encoder_out, xt, t)
        return logits, labels, dec_out, xt
    
    def _get_num_transfer_tokens(self, mask_num, steps):
        """
        Pre-computes the number of tokens to fill at each step of the denoising process
        for a given block, following a linear schedule.
        """
        base = mask_num // steps
        remainder = mask_num % steps
        
        num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_num.device, dtype=torch.long) + base
        
        for i in range(mask_num.size(0)):
            if remainder[i] > 0:
                num_transfer_tokens[i, :remainder[i]] += 1
                
        return num_transfer_tokens
    
    @torch.no_grad()
    def decode(self, encoder_out, decode_steps, logger=None, temperature=0.7, guidance_scale=1.5, block_length=32):
        """ 
        Block-wise Iterative Refinement.
        """
        B = encoder_out.size(0)
        device = encoder_out.device
        max_len = FORMAT_INFO[self.format_name]['max_len']
        
        assert max_len % block_length == 0, "max_len must be divisible by block_length"
        num_blocks = max_len // block_length
        assert decode_steps % num_blocks == 0, "decode_steps must be divisible by num_blocks"
        steps_per_block = decode_steps // num_blocks
        
        # Init fully masked sequence
        xt = torch.full((B, max_len), MASK_ID, dtype=torch.long, device=device)
        xt[:, 0] = SOS_ID # 确保 SOS token 存在
        
        if logger:
            logger.log_inference_start(xt, decode_steps)
            
        log_timesteps = set(np.linspace(0, decode_steps - 1, 240, dtype=int))
        
        timesteps = torch.linspace(1.0, 0.0, decode_steps + 1, device=device)
        
        eos_locked = torch.zeros(B, dtype=torch.bool, device=device)
        eos_position = torch.zeros(B, dtype=torch.long, device=device)
        sample_completed = torch.zeros(B, dtype=torch.bool, device=device)
        eos_threshold = 0.8
        
        early_stop_flag = False
        for block_idx in range(num_blocks):
            if early_stop_flag:
                break
                
            active_end_pos = (block_idx + 1) * block_length
            
            for i in range(steps_per_block):
                if sample_completed.all():
                    early_stop_flag = True
                    break
                
                global_step_idx = block_idx * steps_per_block + i
                t = timesteps[global_step_idx]
                time_batch = torch.full((B,), t, device=device)
                
                active_indices = torch.where(~sample_completed)[0]
                B_active = active_indices.numel()
                
                pred_x0_logits_active = None
                
                if guidance_scale > 1.0:
                    uncond_embedding = self.unconditional_embedding.expand(B, -1, -1)
                    uncond_embedding_active = uncond_embedding[active_indices]
                    encoder_out_active = encoder_out[active_indices]
                    combined_encoder_out = torch.cat([encoder_out_active, uncond_embedding_active], dim=0)
                    
                    xt_active = xt[active_indices]
                    combined_xt = torch.cat([xt_active, xt_active], dim=0)
                    
                    time_batch_active = time_batch[active_indices]
                    combined_t = torch.cat([time_batch_active, time_batch_active], dim=0)
                    
                    combined_logits, _ = self._run_decoder(combined_encoder_out, combined_xt, combined_t)
                    logits_cond, logits_uncond = combined_logits.chunk(2, dim=0)
                    pred_x0_logits_active = logits_uncond + guidance_scale * (logits_cond - logits_uncond)
                else:
                    encoder_out_active = encoder_out[active_indices]
                    xt_active = xt[active_indices]
                    time_batch_active = time_batch[active_indices]
                    pred_x0_logits_active, _ = self._run_decoder(encoder_out_active, xt_active, time_batch_active)
                
                pred_x0_logits = torch.zeros(
                    B, max_len, self.vocab_size, 
                    device=device, 
                    dtype=pred_x0_logits_active.dtype
                )
                pred_x0_logits[active_indices] = pred_x0_logits_active
                
                if temperature > 0.0:
                    pred_x0_probs = torch.softmax(pred_x0_logits / temperature, dim=-1)
                    sampled_tokens = torch.multinomial(pred_x0_probs.view(-1, self.vocab_size), num_samples=1).view(B, max_len)
                else:
                    pred_x0_probs = torch.softmax(pred_x0_logits, dim=-1)
                    sampled_tokens = torch.argmax(pred_x0_probs, dim=-1)
                
                confidence = torch.gather(pred_x0_probs, -1, sampled_tokens.unsqueeze(-1)).squeeze(-1)
                
                eos_probs = pred_x0_probs[..., EOS_ID]
                for b in range(B):
                    if not sample_completed[b] and not eos_locked[b]:
                        if active_end_pos > 1:
                            best_eos_prob, best_eos_pos_offset = eos_probs[b, 1:active_end_pos].max(dim=-1)
                            best_eos_position_b = best_eos_pos_offset + 1
                            
                            if best_eos_prob > eos_threshold:
                                eos_locked[b] = True
                                eos_position[b] = best_eos_position_b
                current_step_in_scope = global_step_idx + 1
                total_steps_for_scope = (block_idx + 1) * steps_per_block
                total_positions_in_scope = active_end_pos - 1
                k_tokens_this_step = math.ceil((current_step_in_scope * total_positions_in_scope) / total_steps_for_scope)
                
                next_xt = torch.full_like(xt, MASK_ID)
                next_xt[:, 0] = SOS_ID
                
                for b in range(B):
                    if sample_completed[b]:
                        next_xt[b] = xt[b]
                        continue
                    if eos_locked[b]:
                        pos_eos = eos_position[b].item()
                        next_xt[b, pos_eos] = EOS_ID
                        if pos_eos < max_len - 1:
                            next_xt[b, pos_eos + 1:] = PAD_ID
                            
                    active_mask_b = torch.zeros(max_len, dtype=torch.bool, device=device)
                    active_mask_b[1:active_end_pos] = True
                    if eos_locked[b]:
                        active_mask_b[eos_position[b]:] = False
                    
                    is_content_pred_b = (sampled_tokens[b] != PAD_ID)
                    valid_for_unmask_b = active_mask_b & is_content_pred_b
                    
                    confidence_masked_b = torch.where(valid_for_unmask_b, confidence[b], -torch.inf)
                    
                    num_locked_in_active = 1 if (eos_locked[b] and eos_position[b] < active_end_pos) else 0
                    k_for_b = k_tokens_this_step - num_locked_in_active
                    
                    num_available_non_pad = valid_for_unmask_b.sum().item()
                    k_for_b = max(0, min(k_for_b, num_available_non_pad))
                    
                    if k_for_b > 0:
                        _, top_indices = torch.topk(confidence_masked_b, k=k_for_b, dim=-1)
                        next_xt[b, top_indices] = sampled_tokens[b, top_indices]
                
                xt = next_xt
                
                for b in range(B):
                    if not sample_completed[b]:
                        if eos_locked[b]:
                            pos_eos = eos_position[b].item()
                            if not (xt[b, 1:pos_eos] == MASK_ID).any():
                                sample_completed[b] = True
                
                if logger and global_step_idx in log_timesteps:
                    logger.log_inference_step(global_step_idx, xt)
        
        predictions = xt
        
        if logger:
            logger.log_inference_step(decode_steps, predictions)
        
        final_preds = []
        for i in range(B):
            pred = predictions[i]
            try:
                eos_idx = (pred == EOS_ID).nonzero(as_tuple=True)[0][0]
                pred = pred[:eos_idx + 1]
            except IndexError:
                pass
            final_preds.append(pred)
            
        dummy_scores = [torch.tensor([0.0]) for _ in range(B)]
        
        t_final = torch.zeros((B,), device=device, dtype=torch.long)
        padded_preds = pad_sequence(final_preds, batch_first=True, padding_value=PAD_ID)
        
        if guidance_scale > 1.0:
            uncond_embedding = self.unconditional_embedding.expand(B, -1, -1)
            
            combined_encoder_out = torch.cat([encoder_out, uncond_embedding], dim=0)
            combined_padded_preds = torch.cat([padded_preds, padded_preds], dim=0)
            combined_t_final = torch.cat([t_final, t_final], dim=0)
            
            _, combined_dec_out = self._run_decoder(combined_encoder_out, combined_padded_preds, combined_t_final)
            dec_out_cond, dec_out_uncond = combined_dec_out.chunk(2, dim=0)
            
            return final_preds, dummy_scores, dummy_scores, (dec_out_cond, dec_out_uncond)
        else:
            _, final_dec_out = self._run_decoder(encoder_out, padded_preds, t_final)
        
        return final_preds, dummy_scores, dummy_scores, (final_dec_out, None)

# WRAPPER and GRAPH PREDICTOR 
class GraphPredictor(nn.Module):
    def __init__(self, decoder_dim, coords=False):
        super(GraphPredictor, self).__init__()
        
        self.lin_i = nn.Linear(decoder_dim, decoder_dim * 2)
        self.lin_j = nn.Linear(decoder_dim, decoder_dim * 2)
        
        self.mlp_rest = nn.Sequential(
            nn.GELU(),
            nn.Linear(decoder_dim * 2, 7) # 0: no bond, 1-6: bond types
        )
        if coords:
            self.coords_mlp = nn.Sequential(
                nn.Linear(decoder_dim, decoder_dim), nn.GELU(),
                nn.Linear(decoder_dim, 2)
            )

    def forward(self, hidden, indices=None):
        """
        Args:
            hidden (Tensor): The decoder's final hidden states, shape [B, L, D].
            indices (Tensor): Padded tensor of atom indices, shape [B, max_num_atoms].
        """
        B, L, D = hidden.size()
        
        counts = (indices != PAD_ID).sum(dim=1)
        max_atoms = counts.max().item()
        
        if max_atoms == 0:
            return {'edges': torch.empty(B, 7, 0, 0, device=hidden.device)}

        batch_idx = torch.arange(B, device=hidden.device).unsqueeze(1).expand(-1, max_atoms)

        valid_indices = indices[:, :max_atoms]
        atom_hidden = hidden[batch_idx, valid_indices] # Shape: [B, max_atoms, D]
        
        arange_mask = torch.arange(max_atoms, device=hidden.device).expand(B, -1)
        padding_mask = arange_mask >= counts.unsqueeze(1)

        atom_hidden[padding_mask] = 0

        h_i = self.lin_i(atom_hidden).unsqueeze(2) # [B, max_atoms, 1, 2D]
        h_j = self.lin_j(atom_hidden).unsqueeze(1) # [B, 1, max_atoms, 2D]
        
        h_pair = h_i + h_j # [B, max_atoms, max_atoms, 2D]
        
        edge_logits = self.mlp_rest(h_pair).permute(0, 3, 1, 2) # Shape: [B, 7, max_atoms, max_atoms]
        
        edge_padding_mask = padding_mask.unsqueeze(1).unsqueeze(3) | padding_mask.unsqueeze(1).unsqueeze(2)
        edge_logits.masked_fill_(edge_padding_mask, -100) # -100 is ignore_index
        
        results = {'edges': edge_logits}
        
        if hasattr(self, 'coords_mlp') and self.coords:
            results['coords'] = self.coords_mlp(atom_hidden)
            
        return results

class Decoder(nn.Module):
    """
    A wrapper for the D3PM-based decoder and the GraphPredictor.
    """
    def __init__(self, args, tokenizer):
        super(Decoder, self).__init__()
        self.args = args
        self.formats = args.formats
        self.tokenizer = tokenizer
        
        input_dim = args.encoder_dim
        num_patches = (args.input_size // 32)**2
        
        self.decoder = nn.ModuleDict()
        
        self.seq_format = next((f for f in ['chartok_coords', 'atomtok_coords'] if f in self.formats), None)
        if self.seq_format is None:
            raise ValueError("No valid sequence format found in args.formats")
        
        self.feature_fusion = FeatureFusion(
            stage3_dim=args.encoder_dim // 2,
            stage4_dim=args.encoder_dim,
        )
        
        self.decoder[self.seq_format] = DiffusionDecoder(args, tokenizer[self.seq_format], format_name=self.seq_format, input_dim=input_dim, num_patches=num_patches)
        
        if 'edges' in self.formats:
            self.decoder['edges'] = GraphPredictor(args.dec_hidden_size, coords=args.continuous_coords)

    
    def forward(self, encoder_out, hiddens, refs):
        """
        Training mode.
        """
        results = {}
        
        x0 = refs[self.seq_format][0] if isinstance(refs[self.seq_format], (tuple, list)) else refs[self.seq_format]
        
        feat_stage3 = hiddens[-2]
        feat_stage4 = hiddens[-1]
        
        if feat_stage3.dim() == 4:
            feat_stage3 = feat_stage3.flatten(1, 2)
        if feat_stage4.dim() == 4:
            feat_stage4 = feat_stage4.flatten(1, 2)
            
        encoder_out = self.feature_fusion(feat_stage3, feat_stage4)
        
        B = encoder_out.size(0)
        device = encoder_out.device
        
        
        # Apply CFG with the probability of p_uncond
        if self.decoder[self.seq_format].p_uncond > 0:
            uncond_mask = (torch.rand(B, device=device) < self.decoder[self.seq_format].p_uncond)
            uncond_embedding = self.decoder[self.seq_format].unconditional_embedding.expand(B, -1, -1)
            encoder_out = torch.where(uncond_mask.view(B, 1, 1), uncond_embedding, encoder_out)
        
        # Get predicted x0 logits and the final decoder hidden states
        logits, labels, dec_out, xt = self.decoder[self.seq_format](encoder_out, x0)
        
        # Package results for the loss function
        results[self.seq_format] = (logits, x0, xt, labels, dec_out)
        
        if 'edges' in self.formats:
            atom_indices = refs['atom_indices'][0] if isinstance(refs['atom_indices'], (tuple, list)) else refs['atom_indices']
            results['edges'] = self.decoder['edges'](dec_out, atom_indices)
            
        return results

    def decode(self, encoder_out, hiddens, decode_steps, logger=None, temperature=0.7, guidance_scale=1.5, block_length=32):
        """
        Inference mode.
        """
        
        feat_stage3 = hiddens[-2]
        feat_stage4 = hiddens[-1]
        
        if feat_stage3.dim() == 4:
            feat_stage3 = feat_stage3.flatten(1, 2)
        if feat_stage4.dim() == 4:
            feat_stage4 = feat_stage4.flatten(1, 2)
            
        encoder_out = self.feature_fusion(feat_stage3, feat_stage4)
        
        seq_preds, _, _, (dec_out_cond, dec_out_uncond) = self.decoder[self.seq_format].decode(encoder_out, decode_steps, logger, temperature, guidance_scale, block_length)
        
        predictions = []
        tokenizer = self.tokenizer[self.seq_format]
        for i, seq in enumerate(seq_preds):
            pred_dict = tokenizer.sequence_to_smiles(seq.tolist())
            if 'indices' in pred_dict and pred_dict['indices']:
                 pred_dict['atom_indices'] = torch.LongTensor(pred_dict['indices']).to(encoder_out.device)
            else: # Handle cases where no atoms are found
                 pred_dict['atom_indices'] = torch.LongTensor([]).to(encoder_out.device)
                 
            predictions.append({self.seq_format: pred_dict})
            
        if 'edges' in self.formats:
            atom_indices = pad_sequence([p[self.seq_format]['atom_indices'] for p in predictions], batch_first=True, padding_value=PAD_ID)
            
            if atom_indices.numel() > 0:
                edge_preds_cond = self.decoder['edges'](dec_out_cond, atom_indices)['edges']
                
                if guidance_scale > 1.0 and dec_out_uncond is not None:
                    edge_preds_uncond = self.decoder['edges'](dec_out_uncond, atom_indices)['edges']
                    edge_logits = edge_preds_uncond + guidance_scale * (edge_preds_cond - edge_preds_uncond)
                    
                else:
                    edge_logits = edge_preds_cond
                
                edge_probs = F.softmax(edge_logits.permute(0, 2, 3, 1), dim=-1).cpu().numpy()

                for i in range(len(predictions)):
                    num_atoms = len(predictions[i][self.seq_format]['symbols'])
                    if num_atoms > 0:
                        prob = edge_probs[i, :num_atoms, :num_atoms, :]
                        edge_pred, _ = get_edge_prediction(prob)
                        predictions[i]['edges'] = edge_pred
                    else:
                        predictions[i]['edges'] = []
            else:
                 for i in range(len(predictions)):
                    predictions[i]['edges'] = []
                    
        for i in range(len(predictions)):
            if 'atom_indices' in predictions[i][self.seq_format]:
                del predictions[i][self.seq_format]['atom_indices']
        
        return predictions


def get_edge_prediction(edge_prob):
    if len(edge_prob) == 0:
        return [], []
    n = len(edge_prob)
    
    edge_prob_copy = np.copy(edge_prob)

    for i in range(n):
        for j in range(i + 1, n):
            # Symmetrize non-stereo bonds
            for k in range(5):
                avg_prob = (edge_prob_copy[i, j, k] + edge_prob_copy[j, i, k]) / 2
                edge_prob[i, j, k] = avg_prob
                edge_prob[j, i, k] = avg_prob
            # Symmetrize stereo bonds (cross-wise)
            avg_prob_5 = (edge_prob_copy[i, j, 5] + edge_prob_copy[j, i, 6]) / 2
            avg_prob_6 = (edge_prob_copy[i, j, 6] + edge_prob_copy[j, i, 5]) / 2
            edge_prob[i, j, 5] = avg_prob_5
            edge_prob[j, i, 6] = avg_prob_5
            edge_prob[i, j, 6] = avg_prob_6
            edge_prob[j, i, 5] = avg_prob_6
    
    prediction = np.argmax(edge_prob, axis=-1).tolist()
    score = np.max(edge_prob, axis=-1).tolist()
    return prediction, score