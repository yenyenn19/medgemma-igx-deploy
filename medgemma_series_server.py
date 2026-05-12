#!/usr/bin/env python3
"""
MedGemma Series Analyzer (Proxy Mode)
Analyzes entire DICOM series (multiple slices) with one request.
Inference is forwarded to gemma4-api-server running on the same machine.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
import requests
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import numpy as np
import json
import os

app = Flask(__name__)
CORS(app)

GEMMA_API_URL = os.environ.get("GEMMA_API_URL", "http://localhost:5000")

# =============================================================================
# DICOM Processing Functions
# =============================================================================

def convert_slice_to_image(pixel_array_2d, ds=None):
    pixel_array = pixel_array_2d.copy()

    if ds and hasattr(ds, 'WindowCenter') and hasattr(ds, 'WindowWidth'):
        try:
            pixel_array = apply_voi_lut(pixel_array, ds)
        except:
            pass

    pixel_array = pixel_array.astype(float)
    pixel_min, pixel_max = pixel_array.min(), pixel_array.max()

    if pixel_max > pixel_min:
        pixel_array = ((pixel_array - pixel_min) / (pixel_max - pixel_min) * 255)

    pixel_array = pixel_array.astype(np.uint8)

    if ds and hasattr(ds, 'PhotometricInterpretation'):
        if ds.PhotometricInterpretation == "MONOCHROME1":
            pixel_array = 255 - pixel_array

    image = Image.fromarray(pixel_array, mode='L')
    return image.convert('RGB')


def fetch_and_convert_instance(instance_id, orthanc_url):
    url = f"{orthanc_url}/instances/{instance_id}/file"
    response = requests.get(url, timeout=30)

    if response.status_code != 200:
        raise Exception(f"Failed to fetch instance: HTTP {response.status_code}")

    ds = pydicom.dcmread(io.BytesIO(response.content), force=True)

    if not hasattr(ds, 'file_meta') or not hasattr(ds.file_meta, 'TransferSyntaxUID'):
        ds.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian

    if not hasattr(ds, 'PixelData'):
        raise ValueError(f"Instance {instance_id} has no pixel data")

    pixel_array = ds.pixel_array

    if len(pixel_array.shape) == 3:
        pixel_array = pixel_array[0]

    image = convert_slice_to_image(pixel_array, ds)
    instance_num = ds.get('InstanceNumber', 0)

    return image, instance_num

# =============================================================================
# Series Management Functions
# =============================================================================

def get_series_instances(series_id, orthanc_url):
    print("📋 Fetching instance metadata for sorting...")

    # Get all instance IDs first
    response = requests.get(f"{orthanc_url}/series/{series_id}")
    if response.status_code != 200:
        raise Exception(f"Failed to get series: HTTP {response.status_code}")

    instances = response.json().get('Instances', [])

    # Use bulk find to get InstanceNumber for all instances in one request
    find_response = requests.post(
        f"{orthanc_url}/tools/find",
        json={
            "Level": "Instance",
            "Query": {"SeriesInstanceUID": ""},
            "ParentSeries": series_id,
            "Expand": True,
            "RequestedTags": ["InstanceNumber"]
        },
        timeout=30
    )

    if find_response.status_code == 200:
        expanded = find_response.json()
        instance_info = []
        for inst in expanded:
            try:
                inst_num = int(inst.get('RequestedTags', {}).get('InstanceNumber', 999999))
                instance_info.append({'id': inst['ID'], 'number': inst_num})
            except:
                instance_info.append({'id': inst['ID'], 'number': 999999})
    else:
        print("   (sorting unavailable, using default order)")
        instance_info = [{'id': i, 'number': idx} for idx, i in enumerate(instances)]

    instance_info.sort(key=lambda x: x['number'])
    sorted_instances = [info['id'] for info in instance_info]

    print(f"✓ Sorted {len(sorted_instances)} instances by InstanceNumber")
    return sorted_instances


def select_key_instances(instances, num_slices=5):
    total = len(instances)

    if num_slices >= total or num_slices >= 999:
        print(f"📌 Analyzing ALL {total} slices")
        return instances

    step = (total - 1) / (num_slices - 1)
    indices = [int(i * step) for i in range(num_slices)]
    selected = [instances[i] for i in indices]

    print(f"📌 Selected indices: {indices}")
    print(f"   (Even anatomical coverage from start to end)")

    return selected

# =============================================================================
# AI Inference (Proxy to gemma4-api-server)
# =============================================================================

def analyze_slice(image, slice_num, total_slices, prompt):
    if total_slices > 1:
        full_prompt = f"Slice {slice_num} of {total_slices}:\n{prompt}"
    else:
        full_prompt = prompt

    img_bytes = io.BytesIO()
    image.save(img_bytes, format="JPEG", quality=95)
    img_bytes.seek(0)

    files = {
        "image_file": ("slice.jpg", img_bytes, "image/jpeg"),
    }
    data = {
        "text": full_prompt,
        "max_new_tokens": "512",
        "do_sample": "false",
    }

    response = requests.post(
        f"{GEMMA_API_URL}/image/stream",
        files=files,
        data=data,
        stream=True,
        timeout=120,
    )

    if response.status_code != 200:
        raise Exception(f"gemma4-api-server error: HTTP {response.status_code} - {response.text[:200]}")

    result_tokens = []
    for raw_line in response.iter_lines():
        if not raw_line:
            continue
        line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line

        if line.startswith("event:"):
            continue

        if line.startswith("data:"):
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
                token = chunk.get("text", "")
                if token:
                    result_tokens.append(token)
            except json.JSONDecodeError:
                if payload:
                    result_tokens.append(payload)

    return "".join(result_tokens)

# =============================================================================
# Report Generation
# =============================================================================

def synthesize_report(results, series_id, total_instances):
    report = "COMPREHENSIVE SERIES ANALYSIS\n"
    report += f"Series ID: {series_id}\n"
    report += f"Total slices in series: {total_instances}\n"
    report += f"Slices analyzed: {len(results)}\n"
    report += f"\n{'='*60}\n\n"
    report += "FINDINGS BY SLICE:\n\n"

    for result in results:
        report += f"Instance {result['instance_number']}:\n"
        report += f"{result['analysis']}\n\n"
        report += "-" * 60 + "\n\n"

    report += f"{'='*60}\n"
    report += "END OF SERIES ANALYSIS\n"

    return report

# =============================================================================
# Flask Routes
# =============================================================================

@app.route('/health', methods=['GET'])
def health():
    try:
        r = requests.get(f"{GEMMA_API_URL}/health", timeout=5)
        gemma_status = r.json().get("status", "unknown") if r.status_code == 200 else "unreachable"
    except:
        gemma_status = "unreachable"

    return jsonify({
        "status": "healthy",
        "mode": "proxy",
        "gemma_api_url": GEMMA_API_URL,
        "gemma_api_status": gemma_status
    }), 200


@app.route('/series/predict', methods=['POST'])
def series_predict():
    try:
        data = request.json

        series_id = data.get('series_id')
        prompt = data.get('prompt', 'Describe any abnormalities.')
        num_slices = data.get('num_slices', 5)
        orthanc_url = data.get('orthanc_url', 'http://localhost:8042')

        if not series_id:
            return jsonify({"error": "series_id is required"}), 400

        print(f"\n{'='*60}")
        print(f"Analyzing Series: {series_id}")
        print(f"{'='*60}")

        print("📋 Fetching series information...")
        all_instances = get_series_instances(series_id, orthanc_url)
        total_instances = len(all_instances)

        if total_instances == 0:
            return jsonify({"error": "Series has no instances"}), 400

        print(f"✓ Found {total_instances} instances")

        key_instances = select_key_instances(all_instances, num_slices)
        print(f"📌 Will analyze {len(key_instances)} slices")

        results = []

        for i, instance_id in enumerate(key_instances, 1):
            print(f"\n🔍 Processing {i}/{len(key_instances)}...")
            print(f"   Instance ID: {instance_id[:16]}...")

            image, instance_num = fetch_and_convert_instance(instance_id, orthanc_url)
            print(f"   Instance number: {instance_num}")

            print(f"   Analyzing...")
            analysis = analyze_slice(image, instance_num, total_instances, prompt)

            results.append({
                'instance_id': instance_id,
                'instance_number': instance_num,
                'analysis': analysis
            })

            print(f"   ✓ Complete")

        results.sort(key=lambda x: x['instance_number'])
        report = synthesize_report(results, series_id, total_instances)

        print(f"\n✓ Series analysis complete!")

        return jsonify({
            "predictions": [{
                "content": report,
                "series_id": series_id,
                "total_slices": total_instances,
                "slices_analyzed": len(results),
                "analyzed_instances": [r['instance_id'] for r in results]
            }]
        }), 200

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("MedGemma DICOM Series Analyzer (Proxy Mode)")
    print("=" * 60)
    print(f"\nForwarding inference to: {GEMMA_API_URL}")
    print("\nEndpoints:")
    print("  POST /series/predict")
    print("  GET  /health")
    print("=" * 60)
    print("\n🚀 Server starting...\n")

    app.run(host='0.0.0.0', port=8080, debug=False)
