import tkinter as tk
from tkinter import messagebox, filedialog
import sounddevice as sd
import soundfile as sf
from pathlib import Path
import time
import numpy as np
import subprocess
import sys
from cluster_prediction import predict_cluster
from save_cluster import load_cluster

# Audio Settings
fs = 44100
channels = 1

# Recording state
recording = False
audio_buffer = []
start_time = None

BASE_DIR = None
SESSION_DIR = None
last_recording_path = None

LAST_OUTPUT_JSON = None
LAST_OUTPUT_DIR = "alt_pipeline/data/output"
CLUSTER_PKL_PATH = "alt_pipeline/3_cluster.pkl"

WINDOW_LEN_SECONDS = 30
MIN_RECORDING_SECONDS = 60

# Tkinter UI
root = tk.Tk()
root.title("Audio Recorder")
root.geometry("300x550")

# window_len_var = tk.IntVar(value=2)

# Timer Label
timer_label = tk.Label(root, text="00:00", font=("Helvetica", 14))
timer_label.pack(pady=5)

tk.Label(
    root,
    text="Minimum recording length: 1 minute\nWindow length: 30 seconds",
    fg="gray",
    font=("Helvetica", 9)
).pack(pady=3)

# Window length selector
# window_frame = tk.Frame(root)
# window_frame.pack(pady=5)

# tk.Label(window_frame, text="Window length (sec):").pack(side=tk.LEFT)

# window_spinbox = tk.Spinbox(
    # window_frame,
    # from_=1,
    # to=60,
    # increment=1,
    # textvariable=window_len_var,
    # width=5
# )
# window_spinbox.pack(side=tk.LEFT, padx=5)

# Speaker Settings
speaker_frame = tk.LabelFrame(root, text="Speaker Settings")
speaker_frame.pack(pady=5, padx=10, fill="x")

speaker_mode_var = tk.StringVar(value="exact")
num_speakers_var = tk.IntVar(value=2)
min_speakers_var = tk.IntVar(value=1)
max_speakers_var = tk.IntVar(value=5)

# Mode selection
mode_frame = tk.Frame(speaker_frame)
mode_frame.pack(pady=2)

def update_speaker_ui():
    mode = speaker_mode_var.get()
    if mode == "exact":
        range_frame.pack_forget()
        exact_frame.pack(pady=5)
    else:
        exact_frame.pack_forget()
        range_frame.pack(pady=5)

tk.Radiobutton(mode_frame, text="Exact Number", variable=speaker_mode_var, value="exact", command=update_speaker_ui).pack(side=tk.LEFT, padx=5)
tk.Radiobutton(mode_frame, text="Range", variable=speaker_mode_var, value="range", command=update_speaker_ui).pack(side=tk.LEFT, padx=5)

# Exact number frame
exact_frame = tk.Frame(speaker_frame)
tk.Label(exact_frame, text="Speakers:").pack(side=tk.LEFT)
tk.Spinbox(exact_frame, from_=1, to=10, textvariable=num_speakers_var, width=5).pack(side=tk.LEFT, padx=5)

# Range number frame
range_frame = tk.Frame(speaker_frame)
tk.Label(range_frame, text="Min:").pack(side=tk.LEFT)
tk.Spinbox(range_frame, from_=1, to=10, textvariable=min_speakers_var, width=3).pack(side=tk.LEFT, padx=2)
tk.Label(range_frame, text="Max:").pack(side=tk.LEFT)
tk.Spinbox(range_frame, from_=1, to=10, textvariable=max_speakers_var, width=3).pack(side=tk.LEFT, padx=2)

# Initial UI state
update_speaker_ui()

# Start/Stop Button
record_btn = tk.Button(root, text="Start Recording", width=20)
record_btn.pack(pady=5)

# Open Archive Button
open_btn = tk.Button(
    root,
    text="Open Session Archive",
    width=20
)
open_btn.pack(pady=5)

# Run Pipeline Button
pipeline_btn = tk.Button(
    root,
    text="Run Pipeline",
    width=20,
    state=tk.DISABLED
)
pipeline_btn.pack(pady=5)

# Load Features JSON Button
load_json_btn = tk.Button(
    root,
    text="Load Features JSON",
    width=20
)
load_json_btn.pack(pady=5)

# Assign Clusters Button
assign_btn = tk.Button(
    root,
    text="Assign Clusters",
    width=20,
    state=tk.DISABLED
)
assign_btn.pack(pady=5)

# Status label
status_label = tk.Label(
    root,
    text="No session recorded yet.",
    wraplength=260,
    justify="left",
    fg="gray"
)
status_label.pack(pady=5)

# Sounddevice stream callback
def audio_callback(indata, frames, time_info, status):
    if recording:
        audio_buffer.append(indata.copy())

stream = sd.InputStream(
    samplerate=fs,
    channels=channels,
    callback=audio_callback
)

# Timer update
def update_timer():
    if recording and start_time is not None:
        elapsed = int(time.time() - start_time)
        timer_label.config(
            text=f"{elapsed // 60:02d}:{elapsed % 60:02d}"
        )
        root.after(100, update_timer)

def toggle_recording():
    global recording, audio_buffer, start_time
    global BASE_DIR, SESSION_DIR, last_recording_path

    if not recording:
        # Create new session
        session_id = f"session_{int(time.time())}"
        SESSION_DIR = Path(f"alt_pipeline/data/sessions/{session_id}")
        BASE_DIR = SESSION_DIR / "Audio/raw"
        BASE_DIR.mkdir(parents=True, exist_ok=True)

        (SESSION_DIR / "session.log").touch(exist_ok=True)

        # Start recording
        recording = True
        audio_buffer = []
        start_time = time.time()

        record_btn.config(text="Stop Recording")
        pipeline_btn.config(state=tk.DISABLED)

        stream.start()
        update_timer()
        print(f"Recording started ({session_id})")

    else:
        # Stop recording
        recording = False
        stream.stop()
        record_btn.config(text="Start Recording")

        recording_duration = time.time() - start_time

        if recording_duration < MIN_RECORDING_SECONDS:
            audio_buffer = []  # discard recording
            messagebox.showwarning(
                "Recording too short",
                "The recording must be at least 1 minute long.\n"
                "Please record a longer session."
            )
            timer_label.config(text="00:00")
            return

        # Save the audio
        if audio_buffer:
            audio_data = np.concatenate(audio_buffer, axis=0)
            last_recording_path = BASE_DIR / f"recorded_{int(time.time())}.wav"
            sf.write(last_recording_path, audio_data, fs)

            messagebox.showinfo(
                "Saved",
                f"Audio saved to:\n{last_recording_path}"
            )

            pipeline_btn.config(state=tk.NORMAL)

            status_label.config(
                text=f"Last recording:\n{SESSION_DIR.name}",
                fg="gray"
            )
        else:
            messagebox.showwarning("Warning", "No audio recorded!")

        # Reset timer
        timer_label.config(text="00:00")
        print("Recording stopped.")

def run_pipeline():
    if not SESSION_DIR:
        messagebox.showerror("Error", "No session to process.")
        return
    
    pipeline_btn.config(state=tk.DISABLED)
    root.update()

    try:
        output_path = "alt_pipeline/data/output"
        cmd = [
            sys.executable,
            "alt_pipeline/pipeline.py",
            "--input", str(SESSION_DIR),
            # "--output", "alt_pipeline/data/output",
            "--output", output_path,
            # "--window-len", str(window_len_var.get())
            "--window-len", str(WINDOW_LEN_SECONDS)
        ]

        if speaker_mode_var.get() == "exact":
             cmd.extend(["--num-speakers", str(num_speakers_var.get())])
        else:
             cmd.extend(["--min-speakers", str(min_speakers_var.get())])
             cmd.extend(["--max-speakers", str(max_speakers_var.get())])

        subprocess.run(cmd, check=True)

        # Track pipeline output JSON
        session_name = SESSION_DIR.name
        output_json = Path(LAST_OUTPUT_DIR) / f"{session_name}_features.json"

        if not output_json.exists():
            raise FileNotFoundError(
                f"Pipeline output not found: {output_json}"
            )

        global LAST_OUTPUT_JSON
        LAST_OUTPUT_JSON = output_json

        messagebox.showinfo(
            "Pipeline finished",
            "Pipeline completed successfully."
        )

        status_label.config(
            text=(
                f"Last processed session:\n"
                f"{SESSION_DIR.name}\n\n"
                f"Output saved in:\n"
                f"{output_path}"
            ),
            fg="green"
        )

        assign_btn.config(state=tk.NORMAL)

    except subprocess.CalledProcessError as e:
        messagebox.showerror(
            "Pipeline error",
            f"Pipeline failed.\n\n{e}"
        )
        status_label.config(
            text=f"Pipeline failed for:\n{SESSION_DIR.name}",
            fg="red"
        )

    pipeline_btn.config(state=tk.NORMAL)

def open_session_archive():
    global SESSION_DIR, BASE_DIR, last_recording_path

    selected_dir = filedialog.askdirectory(
        title="Select session directory"
    )

    if not selected_dir:
        return

    selected_dir = Path(selected_dir)

    # Basic validation
    if not (selected_dir / "session.log").exists():
        messagebox.showerror(
            "Invalid session",
            "Selected folder does not contain session.log"
        )
        return

    SESSION_DIR = selected_dir
    BASE_DIR = SESSION_DIR / "Audio/raw"

    status_label.config(
        text=f"Loaded session:\n{SESSION_DIR.name}",
        fg="blue"
    )

    pipeline_btn.config(state=tk.NORMAL)
    assign_btn.config(state=tk.DISABLED)

    messagebox.showinfo(
        "Session loaded",
        f"Session loaded successfully:\n{SESSION_DIR}"
    )

def load_features_json():
    global LAST_OUTPUT_JSON, SESSION_DIR

    json_path = filedialog.askopenfilename(
        title="Select pipeline output JSON",
        filetypes=[("JSON files", "*.json")]
    )

    if not json_path:
        return

    json_path = Path(json_path)

    if not json_path.exists():
        messagebox.showerror(
            "Invalid file",
            "Selected JSON file does not exist."
        )
        return

    # Set as latest pipeline output
    LAST_OUTPUT_JSON = json_path

    # Clear session context (optional but clean)
    SESSION_DIR = None

    status_label.config(
        text=(
            "Loaded features JSON:\n"
            f"{json_path.name}"
        ),
        fg="blue"
    )

    assign_btn.config(state=tk.NORMAL)

    messagebox.showinfo(
        "JSON loaded",
        f"Feature file loaded:\n{json_path}"
    )

def assign_clusters():
    global LAST_OUTPUT_JSON

    if LAST_OUTPUT_JSON is None or not LAST_OUTPUT_JSON.exists():
        messagebox.showerror(
            "Error",
            "No pipeline output found.\nRun the pipeline first."
        )
        return

    try:
        # 1. Load trained cluster model
        clustered_data = load_cluster(CLUSTER_PKL_PATH)

        # 2. Predict clusters per window
        window_clusters = predict_cluster(
            clustered_data,
            json_path=str(LAST_OUTPUT_JSON)
        )

        # Ensure numpy array
        window_clusters = np.asarray(window_clusters)

        # 3. Session-level cluster (majority vote)
        session_cluster = np.bincount(window_clusters).argmax()

        # 4. Show results
        messagebox.showinfo(
            "Cluster Assignment",
            # f"Session cluster: {session_cluster}\n\n"
            f"Per-window clusters:\n"
            f"{window_clusters.tolist()}"
        )

        source_name = (
            SESSION_DIR.name
            if SESSION_DIR is not None
            else LAST_OUTPUT_JSON.name
        )

        status_label.config(
            text=(
                f"Source: {source_name}\n"
                # f"Assigned cluster: {session_cluster}"
                f"Assigned clusters per window: {window_clusters.tolist()}"
            ),
            fg="purple"
        )

    except Exception as e:
        messagebox.showerror(
            "Cluster assignment failed",
            str(e)
        )

# Bind buttons
record_btn.config(command=toggle_recording)
pipeline_btn.config(command=run_pipeline)
assign_btn.config(command=assign_clusters)
open_btn.config(command=open_session_archive)
load_json_btn.config(command=load_features_json)

# Run UI
root.mainloop()