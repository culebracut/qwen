from qwen_tts import Qwen3TTSModel
import torch
import gc

def run_voice_design(model_path, persona_description):
    # 1. Load Design Model
    model = Qwen3TTSModel.from_pretrained(
        model_path,
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    )
    
    # 2. Run the specific design method
    with torch.no_grad():
        # Using your specific method here
        ref_wavs, sr = model.generate_voice_design(
            description=persona_description,
            # ... other args ...
        )
        # MOVE TO CPU: Essential to prevent memory leaking out of function
        ref_wavs = ref_wavs.detach().cpu()
    
    # 3. Clean up inside
    del model
    return ref_wavs, sr

def run_voice_clone(model_path, ref_audio_path, target_text):
    # 1. Load Clone/Base Model
    model = Qwen3TTSModel.from_pretrained(
        model_path,
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    )
    
    # 2. Run the specific clone method
    with torch.no_grad():
        # Using your second specific method here
        clone_prompt = model.create_voice_clone_prompt(
            audio_path=ref_audio_path,
            text=target_text
        )
        clone_prompt = clone_prompt.detach().cpu()
        
    del model
    return clone_prompt

# --- Execution Flow ---

# Step 1: Design the voice
design_audio, sampling_rate = run_voice_design("/workspace/qwen/models/...VoiceDesign", "Indian Hindu Woman")

# Step 2: Clear GPU completely
gc.collect()
torch.cuda.empty_cache()

# Step 3: Use the Clone model
final_prompt = run_voice_clone("/workspace/qwen/models/...Base", "/workspace/audio/input/...", "Target text")
