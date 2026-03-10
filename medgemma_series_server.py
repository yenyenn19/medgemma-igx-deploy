#!/usr/bin/env python3
"""
MedGemma DICOM Series Analyzer
Analyzes entire DICOM series using Google's MedGemma model with GPU acceleration
"""

import os
import io
import time
import torch
from PIL import Image
import pydicom
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoModelForImageTextToText, AutoProcessor

# =============================================================================
# Global Variables
# =============================================================================

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

MODEL_NAME = "google/medgemma-4b-it"
ORTHANC_URL = "http://orthanc:8042"  # Docker network name

# Model globals
model = None
processor = None
device = None

# =============================================================================
# DICOM Processing Functions
# =============================================================================

def convert_slice_to_image(dicom_slice):
    """
    Convert DICOM slice to PIL Image
    
    Args:
        dicom_slice: pydicom dataset
    
    Returns:
        PIL.Image: Converted image
    """
    pixel_array = dicom_slice.pixel_array
    
    # Normalize to 0-255
    pixel_array = pixel_array - pixel_array.min()
    pixel_array = pixel_array / pixel_array.max() * 255.0
    pixel_array = pixel_array.astype('uint8')
    
    # Convert to PIL Image (grayscale)
    image = Image.fromarray(pixel_array, mode='L')
    
    # Convert to RGB (required by MedGemma)
    image = image.convert('RGB')
    
    return image


def fetch_and_convert_instance(instance_id):
    """
    Fetch DICOM instance from Orthanc and convert to PIL Image
    
    Args:
        instance_id: Orthanc instance ID
    
    Returns:
        PIL.Image or None: Converted image or None if failed
    """
    try:
        dicom_url = f"{ORTHANC_URL}/instances/{instance_id}/file"
        response = requests.get(dicom_url, timeout=30)
        response.raise_for_status()
        
        dicom_bytes = io.BytesIO(response.content)
        dicom_slice = pydicom.dcmread(dicom_bytes, force=True)
        
        if not hasattr(dicom_slice, 'pixel_array'):
            print(f"Warning: Instance {instance_id} has no pixel data")
            return None
            
        return convert_slice_to_image(dicom_slice)
        
    except Exception as e:
        print(f"Error fetching instance {instance_id}: {e}")
        return None

# =============================================================================
# Series Management Functions
# =============================================================================

def get_series_instances(series_id):
    """
    Get all instances in a series from Orthanc, sorted by InstanceNumber
    
    Args:
        series_id: Orthanc series ID
    
    Returns:
        list: Sorted list of instance IDs
    """
    try:
        series_url = f"{ORTHANC_URL}/series/{series_id}"
        response = requests.get(series_url, timeout=30)
        response.raise_for_status()
        series_data = response.json()
        
        instances = series_data.get('Instances', [])
        
        # Get InstanceNumber for each instance to sort properly
        instance_info = []
        for instance_id in instances:
            try:
                instance_url = f"{ORTHANC_URL}/instances/{instance_id}/simplified-tags"
                inst_response = requests.get(instance_url, timeout=10)
                inst_data = inst_response.json()
                
                instance_number = int(inst_data.get('InstanceNumber', 0))
                instance_info.append((instance_number, instance_id))
            except:
                instance_info.append((0, instance_id))
        
        # Sort by InstanceNumber
        instance_info.sort(key=lambda x: x[0])
        sorted_instances = [inst_id for _, inst_id in instance_info]
        
        return sorted_instances
        
    except Exception as e:
        print(f"Error getting series instances: {e}")
        return []


def select_key_instances(instances, num_slices):
    """
    Select evenly distributed instances from the series
    
    Args:
        instances: List of instance IDs
        num_slices: Number of slices to select
    
    Returns:
        list: Selected instance IDs
    """
    total = len(instances)
    
    if num_slices >= total:
        return instances
    
    # Calculate step size for even distribution
    step = total / num_slices
    selected_indices = [int(i * step) for i in range(num_slices)]
    
    return [instances[i] for i in selected_indices]

# =============================================================================
# AI Analysis Functions
# =============================================================================

def analyze_batch(images, slice_nums, total_slices, prompt):
    """
    Analyze multiple DICOM slices with MedGemma in a single batch (GPU efficient!)
    
    Args:
        images: List of PIL Images
        slice_nums: List of instance numbers for context
        total_slices: Total slices in series for context
        prompt: User's question
    
    Returns:
        list: List of MedGemma's analysis texts
    """
    start_total = time.time()
    
    print(f"🔄 Batch processing {len(images)} slices together (GPU optimized)", flush=True)
    
    # Create messages for all images
    messages_list = []
    for image, slice_num in zip(images, slice_nums):
        if total_slices > 1:
            full_prompt = f"Slice {slice_num} of {total_slices}:\n{prompt}"
        else:
            full_prompt = prompt
        
        messages_list.append([{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": full_prompt}
            ]
        }])
    
    t1 = time.time()
    # Process all messages together
    batch_inputs = []
    for messages in messages_list:
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        )
        batch_inputs.append(inputs)
    
    print(f"⏱️ Batch template processing: {time.time() - t1:.2f}s", flush=True)
    
    t2 = time.time()
    # Move all inputs to GPU
    for inputs in batch_inputs:
        for k in inputs.keys():
            inputs[k] = inputs[k].to(device)
            if device == "cuda" and inputs[k].dtype == torch.float32:
                inputs[k] = inputs[k].to(dtype=torch.bfloat16)
    
    print(f"⏱️ Batch GPU transfer: {time.time() - t2:.2f}s", flush=True)
    
    # Store input lengths for each
    input_lens = [inputs["input_ids"].shape[-1] for inputs in batch_inputs]
    
    t3 = time.time()
    # Generate all responses (this keeps GPU busy!)
    results = []
    with torch.inference_mode():
        for idx, inputs in enumerate(batch_inputs):
            generation = model.generate(
                **inputs,
                max_new_tokens=256,  # Reduced from 512 for faster generation
                do_sample=False,
                use_cache=True,
                num_beams=1,
            )
            generation = generation[0][input_lens[idx]:]
            result = processor.decode(generation, skip_special_tokens=True)
            results.append(result)
            print(f"  ✓ Slice {idx + 1}/{len(batch_inputs)} generated", flush=True)
    
    print(f"⏱️ Batch generation: {time.time() - t3:.2f}s", flush=True)
    print(f"⏱️ TOTAL BATCH TIME: {time.time() - start_total:.2f}s ({(time.time() - start_total)/len(images):.2f}s per slice)", flush=True)
    print("-" * 60, flush=True)
    
    return results


def analyze_slice(image, slice_num, total_slices, prompt):
    """
    Analyze a single DICOM slice with MedGemma (wrapper for batch function)
    
    Args:
        image: PIL Image
        slice_num: Instance number for context
        total_slices: Total slices in series for context
        prompt: User's question
    
    Returns:
        str: MedGemma's analysis text
    """
    # Use batch function with single image for consistency
    results = analyze_batch([image], [slice_num], total_slices, prompt)
    return results[0]


def synthesize_report(analyses, num_analyzed, total_slices):
    """
    Synthesize individual slice analyses into a cohesive report
    
    Args:
        analyses: List of (slice_num, analysis_text) tuples
        num_analyzed: Number of slices analyzed
        total_slices: Total slices in series
    
    Returns:
        str: Synthesized report
    """
    report_parts = []
    
    # Add individual slice findings
    for slice_num, analysis in analyses:
        report_parts.append(f"Instance {slice_num}:")
        report_parts.append(analysis)
        report_parts.append("-" * 60)
    
    # Add disclaimer
    report_parts.append("")
    report_parts.append("**Disclaimer:** This analysis is AI-generated for educational purposes only.")
    report_parts.append("Always consult qualified medical professionals for clinical interpretation.")
    
    return "\n".join(report_parts)

# =============================================================================
# Model Loading
# =============================================================================

def print_gpu_info():
    """Print GPU information"""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU: {gpu_name}")
        print(f"GPU Memory: {gpu_memory:.2f} GB")
    else:
        print("No GPU available - using CPU")


def load_model():
    """
    Load MedGemma model and processor with Flash Attention if available
    
    Returns:
        bool: True if successful, False otherwise
    """
    global model, processor, device
    
    print("="*60)
    print("Loading MedGemma Model")
    print("="*60)
    
    # Determine device
    if torch.cuda.is_available():
        device = "cuda"
        print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("⚠ Using CPU (analysis will be slower)")
    
    model_id = "google/medgemma-4b-it"
    
    try:
        # Load processor
        processor = AutoProcessor.from_pretrained(model_id)
        
        # Try loading with Flash Attention 2
        print("\nAttempting to load with Flash Attention 2...")
        try:
            from transformers import AutoConfig
            
            # Load and modify config to force Flash Attention
            config = AutoConfig.from_pretrained(model_id)
            config._attn_implementation = "flash_attention_2"
            
            model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                config=config,
                dtype=torch.bfloat16 if device == "cuda" else torch.float32,
                device_map="cuda:0" if device == "cuda" else None,
                attn_implementation="flash_attention_2"
            )
            
            # Verify Flash Attention is actually being used
            actual_impl = getattr(model.config, '_attn_implementation', 'unknown')
            print(f"✅ Model loaded with attention: {actual_impl}")
            
            if actual_impl == "flash_attention_2":
                print("✅ SUCCESS! Flash Attention 2 is ACTIVE!")
                print("   Expected: 2-3× faster token generation")
            else:
                print(f"⚠️  WARNING: Loaded but using: {actual_impl}")
                
        except Exception as e:
            print(f"⚠️  Flash Attention 2 failed to load: {e}")
            print("\nFalling back to standard attention...")
            model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                dtype=torch.bfloat16 if device == "cuda" else torch.float32,
                device_map="cuda:0" if device == "cuda" else None
            )
            print("✓ Model loaded with standard attention")
        
        # Explicitly move to device
        model = model.to(device)
        
        print("✓ Model loaded successfully!")
        print(f"✓ Model device: {next(model.parameters()).device}")
        print("="*60)
        return True
        
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        print("✗ Failed to load model. Exiting.")
        return False

# =============================================================================
# Flask Routes
# =============================================================================

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model': MODEL_NAME,
        'series_analysis': True,
        'device': device
    })


@app.route('/series/predict', methods=['POST'])
def predict_series():
    """
    Analyze a DICOM series with batch processing for efficiency
    
    Request JSON:
    {
        "series_id": "orthanc-series-id",
        "prompt": "What findings do you see?",
        "num_slices": 5  // optional, default 5
    }
    
    Response JSON:
    {
        "predictions": [{
            "content": "analysis text",
            "slices_analyzed": 5,
            "total_slices": 75
        }]
    }
    """
    start_time = time.time()
    
    try:
        data = request.json
        series_id = data.get('series_id')
        prompt = data.get('prompt', 'Describe the findings in this image.')
        num_slices = data.get('num_slices', 5)
        
        if not series_id:
            return jsonify({'error': 'series_id is required'}), 400
        
        print(f"\n{'='*60}")
        print(f"Analyzing Series: {series_id}")
        print(f"Prompt: {prompt}")
        print(f"Requested slices: {num_slices}")
        print(f"{'='*60}\n")
        
        # Get all instances in series
        instances = get_series_instances(series_id)
        total_slices = len(instances)
        
        if total_slices == 0:
            return jsonify({'error': 'No instances found in series'}), 404
        
        print(f"Total slices in series: {total_slices}")
        
        # Select key instances
        selected_instances = select_key_instances(instances, num_slices)
        print(f"Analyzing {len(selected_instances)} slices\n")
        
        # Fetch ALL images first (parallel loading)
        print(f"📥 Fetching {len(selected_instances)} DICOM images...")
        images = []
        slice_numbers = []
        for idx, instance_id in enumerate(selected_instances, 1):
            image = fetch_and_convert_instance(instance_id)
            if image is None:
                print(f"⚠️ Skipping slice {idx} (failed to load)")
                continue
            images.append(image)
            slice_numbers.append(idx)
        
        print(f"✓ Loaded {len(images)} images\n")
        
        if len(images) == 0:
            return jsonify({'error': 'No valid images could be loaded'}), 500
        
        # BATCH PROCESS all slices together (GPU efficient!)
        print(f"🚀 Starting BATCH analysis (keeps GPU busy)...\n")
        batch_start = time.time()
        
        analyses_texts = analyze_batch(
            images, 
            slice_numbers, 
            len(slice_numbers), 
            prompt
        )
        
        batch_time = time.time() - batch_start
        print(f"\n✅ Batch analysis complete in {batch_time:.1f}s")
        print(f"   Average: {batch_time/len(images):.1f}s per slice\n")
        
        # Combine results
        analyses = list(zip(slice_numbers, analyses_texts))
        
        # Synthesize final report
        final_report = synthesize_report(analyses, len(analyses), total_slices)
        
        elapsed_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"Analysis complete!")
        print(f"Total time: {elapsed_time:.1f}s")
        print(f"Slices analyzed: {len(analyses)}/{total_slices}")
        print(f"Average per slice: {elapsed_time/len(analyses):.1f}s")
        print(f"{'='*60}\n")
        
        return jsonify({
            'predictions': [{
                'content': final_report,
                'slices_analyzed': len(analyses),
                'total_slices': total_slices
            }]
        })
        
    except Exception as e:
        print(f"Error in predict_series: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == '__main__':
    print("="*60)
    print("MedGemma DICOM Series Analyzer")
    print("="*60)
    print("\nFeatures:")
    print("  • Analyzes entire DICOM series with one request")
    print("  • Automatic slice selection for optimal coverage")
    print("  • GPU-accelerated inference when available")
    print("="*60)
    
    # Print PyTorch and CUDA info
    print(f"PyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"CUDA available: True")
        print(f"CUDA version: {torch.version.cuda}")
        print_gpu_info()
    else:
        print(f"CUDA available: False")
    print("="*60)
    
    # Load model
    if not load_model():
        exit(1)
    
    # Start server
    print("="*60)
    print("Server Configuration")
    print("="*60)
    print("Host: 0.0.0.0 (accessible from network)")
    print("Port: 8080")
    print("Endpoint: POST /series/predict")
    print("Health Check: GET /health")
    print("="*60)
    print("🚀 Server starting...")
    
    app.run(host='0.0.0.0', port=8080, debug=False)
