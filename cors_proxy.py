#!/usr/bin/env python3
"""
CORS Proxy for Orthanc
Enables browser-based applications to access Orthanc without CORS restrictions
"""

from flask import Flask, request, Response
from flask_cors import CORS
import requests

# =============================================================================
# Configuration
# =============================================================================

ORTHANC_URL = "http://localhost:8042"
PROXY_PORT = 5000
REQUEST_TIMEOUT = 30  # seconds

# =============================================================================
# Flask Application Setup
# =============================================================================

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes and origins

# =============================================================================
# Proxy Routes
# =============================================================================

@app.route('/orthanc/<path:path>', methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'])
def proxy_orthanc(path):
    """
    Proxy all requests to Orthanc with CORS headers
    
    This endpoint forwards requests from the browser to Orthanc,
    adding necessary CORS headers to allow cross-origin access.
    
    Args:
        path: The Orthanc API path to forward to
    
    Returns:
        Response: Proxied response from Orthanc with CORS headers
    
    Methods:
        GET: Retrieve resources
        POST: Create resources
        PUT: Update resources
        DELETE: Delete resources
        OPTIONS: CORS preflight
    """
    # Handle CORS preflight request
    if request.method == 'OPTIONS':
        return '', 204
    
    # Build target URL
    target_url = f"{ORTHANC_URL}/{path}"
    
    try:
        # Forward request based on method
        response = forward_request(request.method, target_url, request)
        
        # Return response with CORS headers
        return create_response(response)
        
    except requests.Timeout:
        return {'error': 'Request to Orthanc timed out'}, 504
    except requests.ConnectionError:
        return {'error': 'Could not connect to Orthanc. Is it running?'}, 503
    except Exception as e:
        return {'error': f'Proxy error: {str(e)}'}, 500


def forward_request(method, url, original_request):
    """
    Forward HTTP request to Orthanc
    
    Args:
        method: HTTP method (GET, POST, PUT, DELETE)
        url: Target URL
        original_request: Original Flask request object
    
    Returns:
        requests.Response: Response from Orthanc
    """
    if method == 'GET':
        return requests.get(
            url,
            params=original_request.args,
            timeout=REQUEST_TIMEOUT
        )
    
    elif method == 'POST':
        return requests.post(
            url,
            json=original_request.json,
            timeout=REQUEST_TIMEOUT
        )
    
    elif method == 'PUT':
        return requests.put(
            url,
            json=original_request.json,
            timeout=REQUEST_TIMEOUT
        )
    
    elif method == 'DELETE':
        return requests.delete(
            url,
            timeout=REQUEST_TIMEOUT
        )


def create_response(orthanc_response):
    """
    Create Flask response with CORS headers
    
    Args:
        orthanc_response: Response object from Orthanc
    
    Returns:
        Response: Flask response with CORS headers
    """
    return Response(
        orthanc_response.content,
        status=orthanc_response.status_code,
        headers={
            'Content-Type': orthanc_response.headers.get('Content-Type', 'application/json'),
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type, Authorization'
        }
    )

# =============================================================================
# Health Check
# =============================================================================

@app.route('/health', methods=['GET'])
def health():
    """
    Health check endpoint
    
    Returns:
        JSON: Status of proxy and Orthanc connectivity
    """
    try:
        # Check if Orthanc is accessible
        response = requests.get(f"{ORTHANC_URL}/system", timeout=5)
        orthanc_reachable = response.status_code == 200
        
        return {
            'status': 'healthy',
            'proxy': 'running',
            'orthanc': 'reachable' if orthanc_reachable else 'unreachable',
            'orthanc_url': ORTHANC_URL
        }, 200
        
    except:
        return {
            'status': 'degraded',
            'proxy': 'running',
            'orthanc': 'unreachable',
            'orthanc_url': ORTHANC_URL
        }, 503

# =============================================================================
# Main Entry Point
# =============================================================================

def print_startup_info():
    """Print startup information"""
    print("="*60)
    print("CORS Proxy Server for Orthanc")
    print("="*60)
    print("\nConfiguration:")
    print(f"  Proxy Port:    {PROXY_PORT}")
    print(f"  Orthanc URL:   {ORTHANC_URL}")
    print(f"  Timeout:       {REQUEST_TIMEOUT}s")
    print("\nEndpoints:")
    print(f"  Proxy:        http://localhost:{PROXY_PORT}/orthanc/<path>")
    print(f"  Health Check: http://localhost:{PROXY_PORT}/health")
    print("\nUsage in HTML:")
    print(f"  const ORTHANC_URL = 'http://localhost:{PROXY_PORT}/orthanc';")
    print("="*60)


if __name__ == '__main__':
    print_startup_info()
    
    print("\n🚀 Starting proxy server...\n")
    
    app.run(
        host='0.0.0.0',
        port=PROXY_PORT,
        debug=False
    )
