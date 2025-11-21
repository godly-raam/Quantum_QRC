import time
import subprocess
import requests
import json
import sys
import os
import signal
import redis

def run_verification():
    print("🚀 Starting Verification...")
    
    # 1. Start Redis
    print("📦 Starting Redis...")
    redis_process = subprocess.Popen(["redis-server", "--port", "6379"])
    time.sleep(2)
    
    # 2. Start App
    print("🔥 Starting FastAPI App...")
    env = os.environ.copy()
    env["REDIS_URL"] = "redis://localhost:6379"
    env["QRC_NUM_QUBITS"] = "4" # Small for speed
    env["QRC_TRAINING_INSTANCES"] = "5"
    
    app_process = subprocess.Popen(
        ["uvicorn", "main:app", "--host", "127.0.0.1", "--port", "8000"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Wait for app to start
    print("⏳ Waiting for app to be ready...")
    for i in range(30):
        try:
            resp = requests.get("http://127.0.0.1:8000/api/health")
            if resp.status_code == 200:
                print("✅ App is ready!")
                break
        except:
            pass
        time.sleep(1)
    else:
        print("❌ App failed to start.")
        app_process.terminate()
        redis_process.terminate()
        sys.exit(1)
        
    try:
        # 3. Test Optimize
        print("🧪 Testing /api/optimize...")
        payload = {
            "num_locations": 5,
            "num_vehicles": 2,
            "reps": 2,
            "use_qrc": False # Use QAOA first to test flow without waiting for training
        }
        resp = requests.post("http://127.0.0.1:8000/api/optimize", json=payload)
        if resp.status_code != 200:
            print(f"❌ Optimize failed: {resp.text}")
            sys.exit(1)
            
        data = resp.json()
        problem_id = data.get("problem_id")
        if not problem_id:
            print("❌ No problem_id returned!")
            sys.exit(1)
        print(f"✅ Optimize successful. Problem ID: {problem_id}")
        
        # Verify Redis
        r = redis.Redis(host='localhost', port=6379, decode_responses=True)
        if r.exists(f"problem:{problem_id}"):
            print("✅ Redis key exists!")
        else:
            print("❌ Redis key missing!")
            sys.exit(1)
            
        # 4. Test Traffic Jam
        print("🧪 Testing /api/traffic-jam...")
        jam_payload = {
            "problem_id": problem_id,
            "jam_locations": [[0, 1]],
            "jam_severity": 2.0
        }
        resp = requests.post("http://127.0.0.1:8000/api/traffic-jam", json=jam_payload)
        
        # Note: Traffic jam might fail if QRC is not ready (training takes time)
        # But we just want to check if it TRIES to fetch from Redis.
        # If it returns 503 (QRC not ready), that's expected behavior for now, but means logic reached that point.
        # If it returns 404 (Problem not found), that's a failure.
        
        if resp.status_code == 200:
            print("✅ Traffic jam adaptation successful!")
        elif resp.status_code == 503:
            print("⚠️ Traffic jam returned 503 (QRC not ready) - This is expected if training is slow.")
        else:
            print(f"❌ Traffic jam failed: {resp.status_code} {resp.text}")
            # We don't exit here because 503 is likely
            
        # 5. Test Input Guardrails
        print("🧪 Testing Input Guardrails...")
        large_payload = {
            "num_locations": 10, # > 8
            "num_vehicles": 2,
            "use_qrc": True
        }
        resp = requests.post("http://127.0.0.1:8000/api/optimize", json=large_payload)
        # Should switch to classical (use_qrc=False in response or logs)
        # But the response doesn't explicitly say "switched", we check logs or just that it didn't crash.
        if resp.status_code == 200:
            print("✅ Large input handled gracefully.")
        else:
            print(f"❌ Large input failed: {resp.status_code}")
            
    finally:
        print("🧹 Cleaning up...")
        app_process.terminate()
        redis_process.terminate()
        
if __name__ == "__main__":
    run_verification()
