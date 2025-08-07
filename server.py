from flask import Flask, request, jsonify
import os
import cv2
import numpy as np
from datetime import datetime
import threading
import time

app = Flask(__name__)

# Configuration
SAVE_FOLDER = "received_frames"
MAX_FRAMES = 1000  # Maximum number of frames to store (prevents disk overflow)

# Global variables
frame_counter = 0
server_running = True

def setup_folder():
    """Create the folder for storing frames if it doesn't exist"""
    if not os.path.exists(SAVE_FOLDER):
        os.makedirs(SAVE_FOLDER)
        print(f"Created folder: {SAVE_FOLDER}")
    else:
        print(f"Using existing folder: {SAVE_FOLDER}")

def cleanup_old_frames():
    """Remove old frames if we exceed MAX_FRAMES"""
    global frame_counter
    
    try:
        files = [f for f in os.listdir(SAVE_FOLDER) if f.endswith('.jpg')]
        files.sort()  # Sort by filename (which includes timestamp)
        
        if len(files) > MAX_FRAMES:
            files_to_remove = files[:-MAX_FRAMES]  # Keep only the latest MAX_FRAMES
            for file in files_to_remove:
                os.remove(os.path.join(SAVE_FOLDER, file))
                print(f"Removed old frame: {file}")
    except Exception as e:
        print(f"Error during cleanup: {e}")

@app.route('/video_frame', methods=['POST'])
def receive_frame():
    """Receive video frame and save as image"""
    global frame_counter, server_running
    
    if not server_running:
        return jsonify({'status': 'error', 'message': 'Server is shutting down'}), 503
    
    try:
        # Get frame data from request
        frame_data = request.data
        
        if not frame_data:
            return jsonify({'status': 'error', 'message': 'No frame data received'}), 400
        
        # Convert bytes to numpy array
        nparr = np.frombuffer(frame_data, np.uint8)
        
        # Decode image
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({'status': 'error', 'message': 'Could not decode frame'}), 400
        
        # Generate filename with timestamp and counter
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # microseconds to milliseconds
        filename = f"frame_{timestamp}_{frame_counter:06d}.jpg"
        filepath = os.path.join(SAVE_FOLDER, filename)
        
        # Save frame
        success = cv2.imwrite(filepath, frame)
        
        if success:
            frame_counter += 1
            
            # Get frame info
            height, width, channels = frame.shape
            file_size = os.path.getsize(filepath)
            
            print(f"Saved frame {frame_counter}: {filename} ({width}x{height}, {file_size/1024:.1f}KB)")
            
            # Cleanup old frames if necessary
            if frame_counter % 50 == 0:  # Check every 50 frames
                cleanup_old_frames()
            
            return jsonify({
                'status': 'success',
                'message': 'Frame saved successfully',
                'frame_number': frame_counter,
                'filename': filename,
                'dimensions': f"{width}x{height}",
                'file_size_kb': round(file_size/1024, 1),
                'total_frames_saved': frame_counter
            })
        else:
            return jsonify({'status': 'error', 'message': 'Failed to save frame'}), 500
            
    except Exception as e:
        error_msg = f"Error processing frame: {str(e)}"
        print(error_msg)
        return jsonify({'status': 'error', 'message': error_msg}), 500

@app.route('/shutdown', methods=['POST'])
def shutdown():
    """Shutdown the server"""
    global server_running
    
    server_running = False
    
    print(f"\n=== Server Shutdown Requested ===")
    print(f"Total frames received: {frame_counter}")
    print(f"Frames stored in: {os.path.abspath(SAVE_FOLDER)}")
    
    # Give a moment for the response to be sent
    def delayed_shutdown():
        time.sleep(1)
        print("Server shutting down...")
        os._exit(0)
    
    threading.Thread(target=delayed_shutdown).start()
    
    return jsonify({
        'status': 'success',
        'message': 'Server shutdown initiated',
        'total_frames_received': frame_counter,
        'storage_folder': os.path.abspath(SAVE_FOLDER)
    })

@app.route('/status', methods=['GET'])
def get_status():
    """Get server status"""
    try:
        files = [f for f in os.listdir(SAVE_FOLDER) if f.endswith('.jpg')]
        total_files = len(files)
        
        # Calculate total storage size
        total_size = sum(os.path.getsize(os.path.join(SAVE_FOLDER, f)) for f in files)
        
        return jsonify({
            'status': 'running' if server_running else 'shutting_down',
            'frames_received': frame_counter,
            'files_stored': total_files,
            'storage_folder': os.path.abspath(SAVE_FOLDER),
            'total_storage_mb': round(total_size / (1024*1024), 2),
            'max_frames_limit': MAX_FRAMES
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/', methods=['GET'])
def home():
    """Home page with server info"""
    return f"""
    <html>
    <head><title>Frame Receiver Server</title></head>
    <body>
        <h1>Frame Receiver Server</h1>
        <p><strong>Status:</strong> {'Running' if server_running else 'Shutting Down'}</p>
        <p><strong>Frames Received:</strong> {frame_counter}</p>
        <p><strong>Storage Folder:</strong> {os.path.abspath(SAVE_FOLDER)}</p>
        <h2>Endpoints:</h2>
        <ul>
            <li><code>POST /video_frame</code> - Receive video frames</li>
            <li><code>POST /shutdown</code> - Shutdown server</li>
            <li><code>GET /status</code> - Get server status</li>
        </ul>
        <p><a href="/status">Check Status (JSON)</a></p>
    </body>
    </html>
    """

if __name__ == '__main__':
    print("=== Frame Receiver Server ===")
    print(f"Server will store frames in: {os.path.abspath(SAVE_FOLDER)}")
    print(f"Maximum frames to keep: {MAX_FRAMES}")
    print("Starting server on http://127.0.0.1:4002")
    print("Press Ctrl+C to stop the server")
    print("=" * 40)
    
    # Setup storage folder
    setup_folder()
    
    try:
        app.run(host='127.0.0.1', port=4002, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"Server error: {e}")
    finally:
        print(f"Final frame count: {frame_counter}")
        print(f"Frames stored in: {os.path.abspath(SAVE_FOLDER)}")