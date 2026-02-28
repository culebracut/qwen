import torch
import numpy as np
import soundfile as sf
from qwen_tts import Qwen3TTSModel

# 1. Load the Model
# Ensure bfloat16 for Flash-Attention 2 and specify the implementation
model_id = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = Qwen3TTSModel.from_pretrained(
    model_id, 
    device_map=device,
    dtype=torch.bfloat16, 
    attn_implementation="flash_attention_2")

# 2. Device Placement
# Since the wrapper lacks .to(), we move the internal PyTorch module
# model.model.to("cuda")
# print("Internal model moved to 3060 Ti.")

# 3. Generation (Base Model / Voice Cloning)
# For the Base model, you must provide a reference audio or use the voice clone method
myWav = "data/audio/input/JohnWayne/JW.wav"
myWavInputText = "This that the White Man calls charity is a fine thing for widows and orphans, but no warrior can accept it, for if he does, he is no longer a man and when he is no longer a man, he is nothing and better off dead."
myPrompt = model.create_voice_clone_prompt(
    ref_audio=myWav, # Path to your 5-10s clip
    ref_text=myWavInputText,
    x_vector_only_mode=False
)

print("--- Generating audio ---")
text_prompt = "Slap some bacon on a biscuit and let's go! We're burnin' daylight!"
try:
    # Use the specific method for the Base/Voice-Clone model
    wavs, sr = model.generate_voice_clone(
        text=text_prompt,
        language="English",
        device="cuda",
        voice_clone_prompt=myPrompt
    )

    wav = np.asarray(wavs[0], dtype=np.float32)
    
    # 4. Save to file
    # audio_data is a numpy array returned by the helper
    outputWavPath = "data/audio/output/output.wav"
    sf.write(outputWavPath, wav, samplerate=24000)
    print("--- Success! Saved to output.wav ---")

except Exception as e:
    print(f"Generation failed: {e}")
