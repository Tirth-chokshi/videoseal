#!/usr/bin/env python3
"""
Custom Text Watermarking with VideoSeal
Embeds custom text messages and decodes them back during extraction
"""

import torch
import torchvision
import torchaudio
import videoseal
import argparse
import os
import sys
from videoseal.utils.display import save_video_audio_to_mp4

# Global variable for AudioSeal availability
AUDIOSEAL_AVAILABLE = False
try:
    from audioseal import AudioSeal
    AUDIOSEAL_AVAILABLE = True
except ImportError:
    print("⚠️ AudioSeal not available. Use --video_only for video-only watermarking.")

class CustomWatermarker:
    def __init__(self, device=None):
        self.device = device or torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
        # Load VideoSeal model
        print("📦 Loading VideoSeal model...")
        self.video_model = videoseal.load("videoseal")
        self.video_model.eval().to(self.device)
        
        # Load AudioSeal model if available
        self.audio_model = None
        self.audio_detector = None
        if AUDIOSEAL_AVAILABLE:
            try:
                print("📦 Loading AudioSeal models...")
                self.audio_model = AudioSeal.load_generator("audioseal_wm_16bits")
                self.audio_detector = AudioSeal.load_detector("audioseal_detector_16bits")
                print("✅ AudioSeal models loaded")
            except Exception as e:
                print(f"⚠️ AudioSeal loading failed: {e}")
                # Note: AUDIOSEAL_AVAILABLE is global, don't modify here
    
    def text_to_binary(self, text, max_bits=256):
        """Convert text to binary string"""
        # Convert text to bytes, then to binary
        text_bytes = text.encode('utf-8')
        binary_str = ''.join(format(byte, '08b') for byte in text_bytes)
        
        # Pad or truncate to max_bits
        if len(binary_str) > max_bits:
            binary_str = binary_str[:max_bits]
            print(f"⚠️ Text truncated to {max_bits} bits")
        else:
            binary_str = binary_str.ljust(max_bits, '0')  # Pad with zeros
        
        return binary_str
    
    def binary_to_text(self, binary_tensor):
        """Convert binary tensor back to text"""
        # Convert tensor to binary string
        if isinstance(binary_tensor, torch.Tensor):
            if len(binary_tensor.shape) > 1:
                binary_tensor = binary_tensor[0]  # Remove batch dimension
            binary_str = ''.join([str(int(bit.item() > 0.5)) for bit in binary_tensor])
        else:
            binary_str = binary_tensor
        
        # Convert binary to text
        text_chars = []
        for i in range(0, len(binary_str), 8):
            byte_str = binary_str[i:i+8]
            if len(byte_str) == 8:
                byte_val = int(byte_str, 2)
                if byte_val == 0:  # Null terminator
                    break
                if 32 <= byte_val <= 126:  # Printable ASCII
                    text_chars.append(chr(byte_val))
                elif byte_val < 32:  # Control characters, likely padding
                    break
        
        return ''.join(text_chars).rstrip('\x00')
    
    def embed_text(self, input_path, output_path, video_text, audio_text=None, video_only=False):
        """Embed custom text in video and optionally audio"""
        
        print(f"🎬 Embedding custom watermarks:")
        print(f"   Video text: '{video_text}'")
        if audio_text and not video_only:
            print(f"   Audio text: '{audio_text}'")
        
        # Read input video
        video, audio, info = torchvision.io.read_video(input_path, output_format="TCHW")
        
        if audio.numel() == 0 and not video_only:
            print("⚠️ No audio track found, switching to video-only mode")
            video_only = True
        
        fps = info["video_fps"]
        sample_rate = info.get("audio_fps", 44100)
        
        # Normalize video
        video = video.float() / 255.0
        video = video.to(self.device)
        
        # === VIDEO WATERMARKING ===
        print("💧 Embedding video watermark...")
        
        # Convert text to binary tensor
        video_binary = self.text_to_binary(video_text, max_bits=256)
        video_msg_tensor = torch.tensor([[int(b) for b in video_binary]], dtype=torch.float32).to(self.device)
        
        # Embed video watermark
        with torch.no_grad():
            video_outputs = self.video_model.embed(video, msgs=video_msg_tensor, is_video=True, lowres_attenuation=True)
        
        watermarked_video = video_outputs["imgs_w"]
        
        print(f"✅ Video watermark embedded: '{video_text}' ({len(video_text)} chars)")
        
        # === AUDIO WATERMARKING ===
        watermarked_audio = audio.float()
        audio_msg_tensor = None
        
        if not video_only and AUDIOSEAL_AVAILABLE and self.audio_model and audio_text:
            print("🎵 Embedding audio watermark...")
            
            # Prepare audio
            audio = audio.float()
            audio_16k = torchaudio.transforms.Resample(sample_rate, 16000)(audio)
            
            # Convert to mono if stereo
            if audio_16k.shape[0] > 1:
                audio_16k_mono = torch.mean(audio_16k, dim=0, keepdim=True)
            else:
                audio_16k_mono = audio_16k
            
            # Add batch dimension
            audio_16k_batched = audio_16k_mono.unsqueeze(0)
            
            # Convert audio text to binary (16 bits max for AudioSeal)
            audio_binary = self.text_to_binary(audio_text, max_bits=16)
            audio_msg_tensor = torch.tensor([[int(b) for b in audio_binary]], dtype=torch.int32)
            
            # Embed audio watermark
            with torch.no_grad():
                watermark = self.audio_model.get_watermark(audio_16k_batched, 16000, message=audio_msg_tensor)
            
            watermarked_audio_16k = audio_16k_batched + watermark
            watermarked_audio_16k = watermarked_audio_16k.squeeze(0)
            
            # Restore original channels if needed
            if audio_16k.shape[0] > 1:
                watermarked_audio_16k = watermarked_audio_16k.repeat(audio_16k.shape[0], 1)
            
            # Resample back to original sample rate
            watermarked_audio = torchaudio.transforms.Resample(16000, sample_rate)(watermarked_audio_16k)
            
            print(f"✅ Audio watermark embedded: '{audio_text}' ({len(audio_text)} chars)")
        
        # === SAVE RESULT ===
        print(f"💾 Saving watermarked video...")
        
        save_video_audio_to_mp4(
            video_tensor=watermarked_video.cpu(),
            audio_tensor=watermarked_audio,
            fps=int(fps),
            audio_sample_rate=int(sample_rate),
            output_filename=output_path
        )
        
        # Save text messages for reference
        message_file = output_path.replace('.mp4', '_messages.txt')
        with open(message_file, 'w') as f:
            f.write(f"Video Message: {video_text}\n")
            f.write(f"Video Binary: {video_binary}\n")
            if audio_text and not video_only:
                audio_binary = self.text_to_binary(audio_text, max_bits=16)
                f.write(f"Audio Message: {audio_text}\n")
                f.write(f"Audio Binary: {audio_binary}\n")
        
        print(f"✅ Watermarked video saved: {output_path}")
        print(f"📝 Messages saved: {message_file}")
        
        return True
    
    def extract_text(self, input_path, video_only=False):
        """Extract and decode text from watermarked video"""
        
        print(f"🔍 Extracting watermarks from: {input_path}")
        
        # Read video
        video, audio, info = torchvision.io.read_video(input_path, output_format="TCHW")
        sample_rate = info.get("audio_fps", 44100)
        
        # === VIDEO EXTRACTION ===
        print("🎬 Extracting video watermark...")
        
        video = video.float() / 255.0
        video = video.to(self.device)
        
        with torch.no_grad():
            video_msg_tensor = self.video_model.extract_message(video)
        
        # Decode video message
        video_text = self.binary_to_text(video_msg_tensor)
        
        print(f"📹 Video watermark detected:")
        print(f"   Raw bits: {video_msg_tensor[0][:32]}... (showing first 32 bits)")
        print(f"   Decoded text: '{video_text}'")
        
        # === AUDIO EXTRACTION ===
        audio_text = None
        audio_confidence = None
        
        if not video_only and AUDIOSEAL_AVAILABLE and self.audio_detector and audio.numel() > 0:
            print("🎵 Extracting audio watermark...")
            
            audio = audio.float()
            
            # Add batch dimension if needed
            if len(audio.shape) == 2:
                audio = audio.unsqueeze(0)
            
            # Convert to mono if stereo
            if audio.shape[1] > 1:
                audio = torch.mean(audio, dim=1, keepdim=True)
            
            # Detect audio watermark
            with torch.no_grad():
                result, audio_msg_tensor = self.audio_detector.detect_watermark(
                    torchaudio.transforms.Resample(sample_rate, 16000)(audio), 16000
                )
            
            # Handle different result formats from AudioSeal
            if isinstance(result, torch.Tensor):
                audio_confidence = result[0].item() if len(result.shape) > 0 else result.item()
            else:
                audio_confidence = float(result)
            
            # Decode audio message
            if audio_msg_tensor is not None:
                audio_text = self.binary_to_text(audio_msg_tensor[0])
            
            print(f"🎵 Audio watermark detected:")
            print(f"   Confidence: {audio_confidence:.4f}")
            print(f"   Raw bits: {audio_msg_tensor[0] if audio_msg_tensor is not None else 'None'}")
            print(f"   Decoded text: '{audio_text}'")
        
        return video_text, audio_text, audio_confidence

def main():
    parser = argparse.ArgumentParser(description="Custom Text Watermarking with VideoSeal")
    parser.add_argument("--input", required=True, help="Input video file")
    parser.add_argument("--output", help="Output video file (for embedding)")
    parser.add_argument("--video_text", help="Text to embed in video watermark")
    parser.add_argument("--audio_text", help="Text to embed in audio watermark (max 2 chars)")
    parser.add_argument("--extract", action="store_true", help="Extract watermarks instead of embedding")
    parser.add_argument("--video_only", action="store_true", help="Use video-only watermarking")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"❌ Input file not found: {args.input}")
        return 1
    
    # Initialize watermarker
    watermarker = CustomWatermarker()
    
    if args.extract:
        # Extract mode
        video_text, audio_text, confidence = watermarker.extract_text(args.input, args.video_only)
        
        print(f"\n🎉 Extraction Results:")
        print(f"📹 Video Message: '{video_text}'")
        if audio_text is not None:
            print(f"🎵 Audio Message: '{audio_text}' (confidence: {confidence:.4f})")
        
    else:
        # Embed mode
        if not args.output:
            args.output = args.input.replace('.mp4', '_watermarked.mp4')
        
        if not args.video_text:
            print("❌ --video_text is required for embedding")
            return 1
        
        # Check text length limits
        if len(args.video_text) > 32:
            print(f"⚠️ Video text too long ({len(args.video_text)} chars), will be truncated to 32 chars")
        
        if args.audio_text and len(args.audio_text) > 2:
            print(f"⚠️ Audio text too long ({len(args.audio_text)} chars), will be truncated to 2 chars")
        
        success = watermarker.embed_text(
            args.input, 
            args.output, 
            args.video_text, 
            args.audio_text, 
            args.video_only
        )
        
        if success:
            print(f"\n🎉 Embedding Complete!")
            print(f"📁 Watermarked video: {args.output}")
            print(f"📝 Test extraction: python3 custom_watermark.py --input '{args.output}' --extract")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
