import argparse
import signal
import os
import subprocess
from pathlib import Path
import json

from utils import ( signal_handler,
    extract_audio_features, find_session_directories,
    extract_audio_segment, extract_robot_data_features, compute_speed_for_window )


def main():
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Extract args
    parser = argparse.ArgumentParser(description='Run feature extraction pipeline')
    parser.add_argument('--input', type=str, required=True, help="Folder path with raw audios")
    parser.add_argument('--output', type=str, required=True, help="Folder path for output files")
    parser.add_argument('--window-len', type=int, required=True, help="Length of audio windows in seconds")
    parser.add_argument('--num-speakers', type=int, required=False, help="Number of speakers for diarization (optional)")
    parser.add_argument('--min-speakers', type=int, required=False, help="Minimum number of speakers for diarization (optional)")
    parser.add_argument('--max-speakers', type=int, required=False, help="Maximum number of speakers for diarization (optional)")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    session_dirs = find_session_directories(args.input)
    if len(session_dirs) == 0:
        print(f"No sessions found in {args.input}!")
        exit(1)
        
    print(f"Found {len(session_dirs)} audio files to process")
    print(f"Preprocessing the data...")
    
    for dir in session_dirs:
        audio_dir = f"{dir}/Audio"
        raw_audio_dir = f"{audio_dir}/raw"
        proc_audio_dir = f"{audio_dir}/processed"
        video_dir = f"{dir}/Videos/top_cam"
        
        if not os.path.exists(raw_audio_dir):
            os.makedirs(raw_audio_dir)
        if not os.path.exists(proc_audio_dir):
            os.makedirs(proc_audio_dir)

        has_video = os.path.exists(video_dir)

        # Video + audio session
        if has_video:
            list_videos = os.listdir(video_dir)
            video_name = [f for f in list_videos if f.endswith(".mp4")][0].split(".")[0]

            raw_wav = f"{raw_audio_dir}/{video_name}.wav"
            if not os.path.exists(raw_wav):
                extract_audio_segment(
                    f"{video_dir}/{video_name}.mp4",
                    raw_wav,
                )
        # Audio only session
        else:
            print(f"{dir}: audio-only session (skipping video preprocessing)")
    
        # Convert all raw audio files
        for wav in os.listdir(raw_audio_dir):
            if not wav.endswith(".wav"):
                continue

            in_wav = f"{raw_audio_dir}/{wav}"
            out_wav = f"{proc_audio_dir}/{wav}"

            if not os.path.exists(out_wav):
                subprocess.run(
                    [
                        "bash",
                        "src/data_pipeline/convert_audio.sh",
                        "-o",
                        proc_audio_dir,
                        in_wav,
                    ],
                    check=True,
                )

    # Process audios sequentially
    processed_cnt = 0
    for dir in session_dirs:
        print(f"Processing {dir}...")
        
        processed_audio_dir = f"{dir}/Audio/processed"
        audio_files = [f for f in os.listdir(processed_audio_dir) if f.endswith('.wav')]
        
        if not audio_files:
            print(f"Warning: No WAV files found in {processed_audio_dir}, skipping...")
            continue
            
        audio_name = audio_files[0]
        audio_path = f"{processed_audio_dir}/{audio_name}"

        video_dir = f"{dir}/Videos/top_cam"
        has_video = os.path.exists(video_dir)

        if has_video:
            video_path = f"{video_dir}/{audio_name.split('.')[0]}.mp4"
            print(f"  Found video: {video_path}")
        else:
            video_path = None
            print(f"  No video found, proceeding with audio only")
        
         # Extract robot data features only if video exists
        if has_video:
            robot_data_features = extract_robot_data_features(dir)
            print(f"Extracted robot data for {dir}...")
        else:
            robot_data_features = []
            print(f"{dir}: audio-only session (skipping robot logs)")
        
        # Extract audio features
        if args.num_speakers:
            print(f"  Using fixed number of speakers: {args.num_speakers}")
            audio_features = extract_audio_features(audio_path, args.window_len, num_speakers=args.num_speakers)
        elif args.min_speakers and args.max_speakers:
            print(f"  Using speaker range: {args.min_speakers} to {args.max_speakers}")
            audio_features = extract_audio_features(audio_path, args.window_len, min_speakers=args.min_speakers, max_speakers=args.max_speakers)
        else:
            print(f"  Using automatic speaker count")
            audio_features = extract_audio_features(audio_path, args.window_len)
        print(f"Extracted audio features for {dir}...")
        
        # Extract robot speed features (batch processing for efficiency)
        if has_video:
            print(f"  Computing robot speed for all windows (batch mode)...")
            from utils import batch_compute_speed_for_windows
            
            # Prepare window list for batch processing
            window_list = [{
                "window_index": w["window_index"],
                "window_start": w["window_start"],
                "window_end": w["window_end"]
            } for w in audio_features]
            
            # Batch compute all windows with calibration caching
            robot_speed_features = batch_compute_speed_for_windows(video_path, window_list)
        else:
            # Audio-only session
            robot_speed_features = [{
                "window_index": w["window_index"],
                "window_start": w["window_start"],
                "window_end": w["window_end"],
                "avg_speed_cm_s": None,
                "num_detections": None,
            } for w in audio_features]

        print(f"Extracted robot speed features for {dir}...")
        
        # Save results
        dir_name = Path(dir).stem
        output_file = os.path.join(args.output, f"{dir_name}_features.json")
        
        with open(output_file, 'w') as f:
            json.dump({
                'session': dir_name,
                'base_window_length': args.window_len,
                'num_windows': len(audio_features),
                'audio_features': audio_features,
                'robot_data_features': robot_data_features,
                'robot_speed_features': robot_speed_features
            }, f, indent=2)
        
        print(f"✓ Completed {dir_name} -> {output_file}")
        processed_cnt += 1
    
    print(f"\n{'='*60}")
    print(f"Pipeline completed!")
    print(f"Successfully processed: {processed_cnt}/{len(session_dirs)} files")
    print(f"Output directory: {args.output}")
    print(f"{'='*60}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{'='*60}")
        print(f"Pipeline interrupted by user (Ctrl+C)")
        print(f"{'='*60}")
        exit(130)