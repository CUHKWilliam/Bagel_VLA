

from torch.nn.functional import scaled_dot_product_attention


def flash_attn_var_len(
    packed_query_states,
    merged_key_states,
    merged_value_states, 
    cu_seqlens_q,
    cu_seqlens_k,
    query_lens,
    key_values_lens,
    is_causal=False,
    scale=None,
    enable_gqa=False
):
    
    batch_size = len(query_lens)
    max_seqlen_q = max(query_lens).item()
    max_seqlen_k = max(key_values_lens).item()
    
    # Pad sequences to same length for batch processing
    padded_queries = []
    padded_keys = []
    padded_values = []
    
    for i in range(batch_size):
        start_q, end_q = cu_seqlens_q[i], cu_seqlens_q[i+1]
        start_k, end_k = cu_seqlens_k[i], cu_seqlens_k[i+1]
        
        q_seq = packed_query_states[start_q:end_q]
        k_seq = merged_key_states[start_k:end_k]
        v_seq = merged_value_states[start_k:end_k]
        
        # Pad sequences to max length
        q_padded = torch.nn.functional.pad(q_seq, (0, 0, 0, 0, 0, max_seqlen_q - q_seq.size(0)))
        k_padded = torch.nn.functional.pad(k_seq, (0, 0, 0, 0, 0, max_seqlen_k - k_seq.size(0)))
        v_padded = torch.nn.functional.pad(v_seq, (0, 0, 0, 0, 0, max_seqlen_k - v_seq.size(0)))
        
        padded_queries.append(q_padded)
        padded_keys.append(k_padded)
        padded_values.append(v_padded)
    
    # Stack to create batch tensors
    queries_batch = torch.stack(padded_queries)  # [batch, max_seqlen_q, num_heads, head_dim]
    keys_batch = torch.stack(padded_keys)        # [batch, max_seqlen_k, num_heads, head_dim]
    values_batch = torch.stack(padded_values)    # [batch, max_seqlen_k, num_heads, head_dim]
    
    # Rearrange dimensions: [batch, seq_len, num_heads, head_dim] -> [batch, num_heads, seq_len, head_dim]
    queries_batch = queries_batch.transpose(1, 2)
    keys_batch = keys_batch.transpose(1, 2)
    values_batch = values_batch.transpose(1, 2)
    
    # Create attention mask
    attn_mask = None
    if is_causal:
        attn_mask = generate_batched_causal_mask(query_lens, key_values_lens, max_seqlen_q, max_seqlen_k, queries_batch.device)
    
    # Apply batched attention
    attn_output_batch = F.scaled_dot_product_attention(
        query=queries_batch,
        key=keys_batch,
        value=values_batch,
        attn_mask=attn_mask,
        dropout_p=0.0,
        is_causal=False,
        scale=scale,
    )  # [batch, num_heads, max_seqlen_q, head_dim]
    
    # Convert back and unpack
    attn_output_batch = attn_output_batch.transpose(1, 2)  # [batch, max_seqlen_q, num_heads, head_dim]
    
    # Unpack back to packed format
    packed_attn_output = torch.empty_like(packed_query_states)
    for i in range(batch_size):
        start_q, end_q = cu_seqlens_q[i], cu_seqlens_q[i+1]
        seq_len = end_q - start_q
        packed_attn_output[start_q:end_q] = attn_output_batch[i, :seq_len]
    
    return packed_attn_output

def generate_batched_causal_mask(query_lens, key_values_lens, max_seqlen_q, max_seqlen_k, device):
    """Generate batched causal mask considering actual sequence lengths"""
    batch_size = len(query_lens)
    mask = torch.zeros(batch_size, max_seqlen_q, max_seqlen_k, device=device)
    
    for i in range(batch_size):
        seq_len_q = query_lens[i]
        seq_len_k = key_values_lens[i]
        # Create causal mask for this sequence
        causal_mask = torch.triu(torch.ones(seq_len_q, seq_len_k, device=device), diagonal=1)
        mask[i, :seq_len_q, :seq_len_k] = causal_mask
    
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask.unsqueeze(1)  # Add head dimension: [batch, 1, seq_len_q, seq_len_k]
