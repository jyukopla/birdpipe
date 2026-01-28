import time
import json
import subprocess

DEFAULT_LOCATION = (62.230208, 25.743281)  # Jyväskylä, Finland
MAX_RETRY_TIME = 40  # Max retry window in seconds
RETRY_INTERVAL = 5   # Time between retries
MAX_ACCEPTABLE_ACCURACY = 500  # meters

import subprocess
import json
import time
from recorder_config import (
    GPS_TIMEOUT_SECONDS, GPS_MIN_ACCURACY_METERS, GPS_RETRY_ATTEMPTS, 
    GPS_RETRY_DELAY, DEFAULT_LATITUDE, DEFAULT_LONGITUDE, GPS_REQUIRED
)

def get_gnss_location():
    """
    Get GNSS location from gpsd with enhanced error handling and retry logic.
    
    Returns:
        tuple: (latitude, longitude, gps_fix_quality, horizontal_accuracy, gps_source)
        - gps_source: "gps" (live GPS), "default" (fallback), or "cached" (last known)
    """
    cached_location = None
    
    for attempt in range(GPS_RETRY_ATTEMPTS):
        print(f"GPS attempt {attempt + 1}/{GPS_RETRY_ATTEMPTS}...")
        
        try:
            # Run gpspipe to get GPS data
            result = subprocess.run(
                ['gpspipe', '-w', '-n', '10'], 
                capture_output=True, 
                text=True, 
                timeout=GPS_TIMEOUT_SECONDS
            )
            
            if result.returncode != 0:
                print(f"GPS attempt {attempt + 1} failed: gpspipe returned {result.returncode}")
                if attempt < GPS_RETRY_ATTEMPTS - 1:
                    print(f"Retrying in {GPS_RETRY_DELAY} seconds...")
                    time.sleep(GPS_RETRY_DELAY)
                continue
            
            # Parse GPS data
            for line in result.stdout.strip().split('\n'):
                if not line:
                    continue
                    
                try:
                    gps_data = json.loads(line)
                    
                    # Check for TPV (Time-Position-Velocity) message
                    if gps_data.get('class') == 'TPV':
                        lat = gps_data.get('lat')
                        lon = gps_data.get('lon')
                        mode = gps_data.get('mode', 0)
                        
                        # GPS fix quality: 0=no fix, 1=GPS, 2=DGPS, 3=3D fix
                        if lat is not None and lon is not None and mode >= 2:
                            # Check horizontal accuracy if available
                            h_acc = gps_data.get('eph', float('inf'))  # estimated horizontal position error
                            
                            # Cache this location as a backup
                            cached_location = (lat, lon, mode, h_acc)
                            
                            # Check if accuracy meets our requirements
                            if h_acc <= GPS_MIN_ACCURACY_METERS:
                                print(f"GPS fix obtained: Lat {lat:.6f}, Lon {lon:.6f}, Mode {mode}, Accuracy {h_acc:.1f}m")
                                return lat, lon, mode, h_acc, "gps"
                            else:
                                print(f"GPS fix too inaccurate: {h_acc:.1f}m > {GPS_MIN_ACCURACY_METERS}m threshold")
                                
                except json.JSONDecodeError:
                    continue
                    
        except subprocess.TimeoutExpired:
            print(f"GPS attempt {attempt + 1} timed out after {GPS_TIMEOUT_SECONDS} seconds")
        except Exception as e:
            print(f"GPS attempt {attempt + 1} error: {e}")
        
        # Wait before retry (except on last attempt)
        if attempt < GPS_RETRY_ATTEMPTS - 1:
            print(f"Retrying in {GPS_RETRY_DELAY} seconds...")
            time.sleep(GPS_RETRY_DELAY)
    
    # All GPS attempts failed
    print("All GPS attempts failed.")
    
    # If we have a cached location (even if not accurate enough), use it
    if cached_location:
        lat, lon, mode, h_acc = cached_location
        print(f"Using cached GPS location: Lat {lat:.6f}, Lon {lon:.6f}, Accuracy {h_acc:.1f}m")
        return lat, lon, mode, h_acc, "cached"
    
    # Check if GPS is required
    if GPS_REQUIRED:
        raise Exception("GPS is required but no GPS fix could be obtained. Recording aborted.")
    
    # Fall back to default coordinates
    print(f"Using default coordinates: Lat {DEFAULT_LATITUDE}, Lon {DEFAULT_LONGITUDE}")
    return DEFAULT_LATITUDE, DEFAULT_LONGITUDE, 0, float('inf'), "default"
