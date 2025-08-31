#!/usr/bin/env python3
"""
Working video analysis script using AvalAI's OpenAI-compatible API.
This approach extracts frames and sends them as images to the API.
"""

import os
import base64
import cv2
import time
from openai import OpenAI
from typing import List, Dict, Any

# Configuration
AVALAI_API_KEY = "aa-Fmk9AQbfxC1mZmEI0efUap2RfDpU7mLi67RIez5pmcUmZ7ym"
AVAL_OPENAI_BASE = "https://api.avalai.ir/v1"

def extract_frames_b64(
    video_path: str, fps: float = 3.0, max_frames: int = 40, jpg_quality: int = 95
) -> List[str]:
    """
    Extract frames from video and convert to base64-encoded JPEG strings.
    """
    print(f"📹 Extracting frames from video: {video_path}")
    print(f"⚙️  Settings: {fps} fps, max {max_frames} frames, quality {jpg_quality}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")
    
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / native_fps
    
    print(f"📊 Video info: {total_frames} frames, {native_fps:.1f} fps, {duration:.2f}s duration")
    
    step = max(1, int(round(native_fps / max(fps, 1e-6))))
    frames, i = [], 0
    
    start_time = time.time()
    while len(frames) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            ok2, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality])
            if ok2:
                frames.append(base64.b64encode(jpg.tobytes()).decode("utf-8"))
                if len(frames) % 5 == 0:
                    print(f"📸 Extracted {len(frames)} frames...")
        i += 1
    
    cap.release()
    extraction_time = time.time() - start_time
    print(f"✅ Frame extraction completed in {extraction_time:.2f}s")
    print(f"📸 Total frames extracted: {len(frames)}")
    
    return frames

def make_messages_with_frames(question: str, frames_b64: List[str]) -> List[Dict[str, Any]]:
    """
    Create OpenAI-style chat messages with multiple images.
    """
    content = [{"type": "text", "text": question}]
    
    for i, b64 in enumerate(frames_b64):
        content.append({
            "type": "image_url", 
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
        })
    
    print(f"📝 Created message with {len(frames_b64)} images")
    return [{"role": "user", "content": content}]

def analyze_video_with_frames(
    video_path: str, 
    question: str, 
    model: str = "gemini-2.5-flash-lite",
    fps: float = 3.0, 
    max_frames: int = 40
) -> str:
    """
    Analyze video by extracting frames and sending to AvalAI API.
    """
    print(f"🚀 Starting video analysis with {model}")
    print(f"❓ Question: {question}")
    
    # Step 1: Extract frames
    start_time = time.time()
    frames = extract_frames_b64(video_path, fps, max_frames)
    extraction_time = time.time() - start_time
    
    # Step 2: Create messages
    messages = make_messages_with_frames(question, frames)
    
    # Step 3: Send to API
    print(f"🌐 Sending request to AvalAI API...")
    api_start_time = time.time()
    
    try:
        client = OpenAI(api_key=AVALAI_API_KEY, base_url=AVAL_OPENAI_BASE)
        
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
            max_tokens=1000
        )
        
        api_time = time.time() - api_start_time
        total_time = time.time() - start_time
        
        print(f"✅ API response received in {api_time:.2f}s")
        print(f"⏱️  Total processing time: {total_time:.2f}s")
        
        return response.choices[0].message.content
        
    except Exception as e:
        api_time = time.time() - api_start_time
        total_time = time.time() - start_time
        print(f"❌ API call failed after {api_time:.2f}s")
        print(f"🔍 Error: {str(e)}")
        raise

def main():
    """Main function to run video analysis"""
    # Configuration
    video_path = "/Users/arshiahemmat/Documents/Code/Physics_dataset/videos/1g25_v1.mp4"
    question = "Analyze this physics simulation video. Describe: 1) What is happening step by step, 2) What physics concepts are demonstrated, 3) What forces are acting on the objects, 4) Any numerical values or measurements shown. Be detailed and scientific in your analysis."
    
    # Available models to test (your specific models)
    available_models = [
        "qwen-vl-plus",
        "qwen2.5-vl-7b-instruct",
        "gpt-4.1-nano",
        "gemini-2.0-flash-lite",
        "gpt-4o-mini"
    ]
    
    print("🎬 AvalAI Video Analysis Tool - Multi-Model Testing")
    print("=" * 60)
    print(f"📁 Video: {video_path}")
    print(f"❓ Question: {question}")
    print(f"🤖 Testing {len(available_models)} models: {', '.join(available_models)}")
    print("=" * 60)
    
    # Check if video exists
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        return
    
    # Store results for comparison
    results = {}
    
    # Run analysis with all models
    for i, model in enumerate(available_models, 1):
        try:
            print(f"\n🔍 Testing model {i}/{len(available_models)}: {model}")
            print("-" * 50)
            
            start_time = time.time()
            result = analyze_video_with_frames(
                video_path=video_path,
                question=question,
                model=model,
                fps=3.0,        # Extract more frames
                max_frames=40   # Send more images
            )
            total_time = time.time() - start_time
            
            # Store result
            results[model] = {
                "success": True,
                "result": result,
                "total_time": total_time
            }
            
            print(f"\n📝 Analysis Result ({model}):")
            print("=" * 50)
            print(result)
            print("=" * 50)
            
        except Exception as e:
            print(f"❌ Model {model} failed: {e}")
            results[model] = {
                "success": False,
                "error": str(e),
                "total_time": 0
            }
            continue
    
    # Summary of all results
    print("\n" + "=" * 60)
    print("📊 MULTI-MODEL ANALYSIS SUMMARY")
    print("=" * 60)
    
    successful_models = [m for m, r in results.items() if r["success"]]
    failed_models = [m for m, r in results.items() if not r["success"]]
    
    print(f"✅ Successful models: {len(successful_models)}")
    for model in successful_models:
        time_taken = results[model]["total_time"]
        print(f"   • {model}: {time_taken:.2f}s")
    
    if failed_models:
        print(f"❌ Failed models: {len(failed_models)}")
        for model in failed_models:
            error = results[model]["error"]
            print(f"   • {model}: {error}")
    
    print(f"\n🏁 Analysis completed! Tested {len(available_models)} models.")
    
    print("\n🏁 Analysis completed!")

if __name__ == "__main__":
    main()
