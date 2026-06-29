import subprocess
import time
import sys

def run_services():
    # 1. Start the FastAPI backend
    # We use a non-blocking call to start the server
    api_process = subprocess.Popen(["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"])
    
    # Give the API a moment to spin up
    time.sleep(5)
    
    # 2. Start the Streamlit frontend
    # Note: Streamlit usually uses port 8501, which Render will map to 10000
    ui_process = subprocess.Popen(["streamlit", "run", "app.py", "--server.port", "10000", "--server.address", "0.0.0.0"])
    
    # Keep the script running to monitor processes
    try:
        api_process.wait()
        ui_process.wait()
    except KeyboardInterrupt:
        api_process.terminate()
        ui_process.terminate()

if __name__ == "__main__":
    run_services()
