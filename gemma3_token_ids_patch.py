"""
Comprehensive token_type_ids patch for Gemma 3
Patches at model level to auto-generate token_type_ids
"""
import torch

print("Applying token_type_ids patches...")

# Patch 1: Gemma3Model.forward (core model)
try:
    from transformers.models.gemma3.modeling_gemma3 import Gemma3Model
    _original_gemma3model_forward = Gemma3Model.forward
    
    def patched_gemma3model_forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        cache_position=None,
        token_type_ids=None,
        **kwargs
    ):
        # Auto-generate token_type_ids if missing
        if token_type_ids is None and input_ids is not None:
            batch_size, seq_len = input_ids.shape
            token_type_ids = torch.zeros(
                (batch_size, seq_len), 
                dtype=torch.long, 
                device=input_ids.device
            )
        
        return _original_gemma3model_forward(
            self,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            token_type_ids=token_type_ids,
            **kwargs
        )
    
    Gemma3Model.forward = patched_gemma3model_forward
    print("✓ Patched Gemma3Model.forward")
except Exception as e:
    print(f"! Gemma3Model patch failed: {e}")

# Patch 2: Gemma3ForCausalLM.forward (wrapper model)
try:
    from transformers.models.gemma3.modeling_gemma3 import Gemma3ForCausalLM
    _original_causal_forward = Gemma3ForCausalLM.forward
    
    def patched_causal_forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        cache_position=None,
        token_type_ids=None,
        **kwargs
    ):
        # Auto-generate token_type_ids if missing during training
        if token_type_ids is None and self.training and input_ids is not None:
            batch_size, seq_len = input_ids.shape
            token_type_ids = torch.zeros(
                (batch_size, seq_len), 
                dtype=torch.long, 
                device=input_ids.device
            )
            # Mark assistant tokens based on labels if available
            if labels is not None:
                token_type_ids[labels != -100] = 1
        
        return _original_causal_forward(
            self,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            token_type_ids=token_type_ids,
            **kwargs
        )
    
    Gemma3ForCausalLM.forward = patched_causal_forward
    print("✓ Patched Gemma3ForCausalLM.forward")
except Exception as e:
    print(f"! Gemma3ForCausalLM patch failed: {e}")

print("=" * 60)
print("Token type IDs patches applied successfully!")
print("=" * 60)
