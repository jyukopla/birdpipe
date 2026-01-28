import os
import pandas as pd
import numpy as np
import logging
from classifier import Classifier
from functions import top_preds, calibrate, adjust
from recorder_config import (
    REGION_MODEL_PATHS,
    REGION_CLASS_PATHS,
    PREDICTION_THRESHOLD,
)

# Configure logging
logging.basicConfig(filename="analysis.log", level=logging.ERROR)

# Try to load optional calibration and migration data
try:
    migr_table = np.load("Pred_adjustment/migration_params.npy")
    cal_table = np.load("Pred_adjustment/calibration_params.npy")
except FileNotFoundError:
    migr_table = None
    cal_table = None


def analyze_audio(
    audio_buffer_or_path,
    latitude,
    longitude,
    recording_time_unix_ns,
    day_of_year,
    region="finland"
):
    region = region.lower()
    print(f"[INFO] Region selected: {region}")

    # Get paths for model and class labels
    model_file = REGION_MODEL_PATHS.get(region, REGION_MODEL_PATHS["finland"])
    class_file = REGION_CLASS_PATHS.get(region, REGION_CLASS_PATHS["finland"])

    print(f"[INFO] Using model file: {model_file}")
    print(f"[INFO] Using class file: {class_file}")
    print(f"[INFO] Prediction threshold: {PREDICTION_THRESHOLD}")

    # Load model
    clsf = Classifier(
        path_to_model=model_file,
        sr=48000,
        clip_dur=3.0,
        TFLITE_THREADS=1,
        offset=0,
        dur=0,
    )

    # Load species label data
    sp_list = pd.read_csv(class_file)

    # Check required columns
    for col in ["common_name", "scientific_name", "class"]:
        if col not in sp_list.columns:
            raise ValueError(f"CSV file is missing required column: '{col}'")

    try:
        # Load audio and run classification
        if isinstance(audio_buffer_or_path, str):
            with open(audio_buffer_or_path, "rb") as audio_file:
                preds, timestamps = clsf.classify(audio_file)
        else:
            preds, timestamps = clsf.classify(audio_buffer_or_path)

        # Mute background noise and human class
        preds[0:2] = [0, 0]

        # Calibration (if region supports it)
        if region != "madagascar" and cal_table is not None:
            preds = calibrate(preds, cal_table=cal_table)

        # Get top predictions
        preds, classes, timestamps = top_preds(preds, timestamps, PREDICTION_THRESHOLD)

        # Migration adjustment only for class > 1
        if region != "madagascar" and migr_table is not None:
            try:
                distribution_map_path = "Pred_adjustment/distribution_maps"
                if os.path.isdir(distribution_map_path):
                    valid_indices = [i for i, cls in enumerate(classes) if cls > 1]
                    valid_classes = [classes[i] for i in valid_indices]
                    valid_preds = [preds[i] for i in valid_indices]

                    preds_adjusted = adjust(valid_preds, valid_classes, migr_table, latitude, longitude, day_of_year)

                    # Replace adjusted predictions back
                    for i, idx in enumerate(valid_indices):
                        preds[idx] = preds_adjusted[i]
                else:
                    print(f"[WARNING] Skipping migration adjustment: missing directory '{distribution_map_path}'")
                    logging.warning(f"Migration adjustment skipped: directory '{distribution_map_path}' not found")

            except FileNotFoundError as e:
                print(f"[WARNING] Migration adjustment skipped: {e}")
                logging.warning(f"Migration adjustment skipped: {e}")
            except Exception as e:
                print(f"[WARNING] Migration adjustment failed: {e}")
                logging.warning(f"Migration adjustment failed: {e}")

        detected_species = []

        print("\nDetected species:")
        for i in range(len(preds)):
            if classes[i] > 1 and preds[i] > PREDICTION_THRESHOLD:
                species_class = classes[i]
                probability = round(preds[i], 2)
                utc_unix_timestamp_ns = recording_time_unix_ns
                offset_time_s = timestamps[i]

                # Find the matching row in the class CSV
                row = sp_list.loc[sp_list["class"] == species_class]

                if not row.empty:
                    species_name = row.iloc[0]["common_name"]
                    scientific_name = row.iloc[0]["scientific_name"]
                else:
                    species_name = f"Class {species_class}"
                    scientific_name = "Unknown"

                print(
                    f" - {species_name} ({scientific_name}) [Class {species_class}]: "
                    f"Probability {probability:.2f}, "
                    f"Time {utc_unix_timestamp_ns}, "
                    f"Offset {offset_time_s:.2f}s"
                )

                detected_species.append(
                    (species_class, probability, utc_unix_timestamp_ns, offset_time_s)
                )

        print(f"Recording analyzed! {len(detected_species)} species detected.")
        return detected_species

    except Exception as e:
        logging.exception(f"Error during analysis: {e}")
        print(f"Error during analysis: {e}")
        return []
