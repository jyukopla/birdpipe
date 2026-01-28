import subprocess
import os
import speedtest

from recorder_config import CLOUD_REMOTE_NAME, CLOUD_REMOTE_PATH

MIN_UPLOAD_MBPS = 0.5
GEOJSON_PATH = "/var/data/geojson"
AUDIO_PATH = "/var/data/audiofiles"
LOG_PATH = "/var/data/logs"


def check_upload_speed():
    print("Testing upload speed...")
    upload_speed = 0
    try:
        st = speedtest.Speedtest()
        st.get_best_server()
        upload_speed = st.upload() / (1024 * 1024)
        print(f"Upload speed: {upload_speed:.2f} Mbps")
        return upload_speed >= MIN_UPLOAD_MBPS
    except Exception as e:
        print(f"Speedtest failed at {upload_speed:.2f} Mbps: {e}")
        return False


def move_and_optionally_remove(filepath, filetype):
    print(f"Transferring file: {filepath}")
    try:
        if filetype in ("geojson", "logs"):
            # Kopioidaan geojson- ja log-tiedostot pilveen, säilytetään paikallisesti
            subprocess.run(
                [
                    "rclone",
                    "copy",
                    filepath,
                    f"{CLOUD_REMOTE_NAME}:{CLOUD_REMOTE_PATH}/{filetype}",
                ],
                check=True,
            )
            print(f"Copied {filetype} file to cloud (retained locally): {filepath}")
        else:
            # Siirretään muut tiedostot pilveen ja poistetaan paikallisesti
            subprocess.run(
                [
                    "rclone",
                    "move",
                    filepath,
                    f"{CLOUD_REMOTE_NAME}:{CLOUD_REMOTE_PATH}/{filetype}",
                ],
                check=True,
            )
            print(f"Successfully moved {filepath} to cloud.")
            # Varmistetaan, että tiedosto poistetaan paikallisesti (move yleensä poistaa, mutta varmistetaan)
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                    print(f"Deleted local file: {filepath}")
                except Exception as e:
                    print(f"Failed to delete local file: {filepath}: {e}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to transfer {filepath}: {e}")


def process_directory(directory, filetype):
    print(f"Processing directory: {directory}")
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            if os.path.isfile(filepath):
                move_and_optionally_remove(filepath, filetype)
    else:
        print(f"Directory does not exist: {directory}")


if __name__ == "__main__":
    print("Starting upload service...")
    process_directory(GEOJSON_PATH, "geojson")
    process_directory(AUDIO_PATH, "audiofiles")
    process_directory(LOG_PATH, "logs")
    print("Upload speed too low, aborting.")
