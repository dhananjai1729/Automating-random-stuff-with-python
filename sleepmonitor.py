import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
from scipy import signal
import time
import os
import datetime
import threading
import matplotlib.pyplot as plt
import cv2
# Note: ipywidgets and IPython.display are for notebook UI, not needed in the class itself.

class SleepMonitor:
    def __init__(self, output_dir="sleep_recordings",
                 channels=1,
                 rate=16000,
                 chunk_seconds=1,
                 calibration_time=5,
                 video_recording_enabled_by_default: bool = False): # New parameter

        self.output_dir = output_dir
        self.channels = channels
        self.rate = rate
        self.chunk_seconds = chunk_seconds
        self.calibration_time = calibration_time

        self.background_noise_level = None
        self.recording = False
        self.frames = [] # Audio frames
        self.start_time = None

        # Video related attributes
        self.video_capture = None
        self.video_writer = None
        self.selected_camera_index = -1
        self.video_recording_enabled = video_recording_enabled_by_default # Set from parameter
        self.video_filename = None # Initialize video_filename

        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Get list of available devices
        try:
            self.devices = sd.query_devices()
            # This print statement is UI related, can be removed or handled by a logger in a real app
            # print("Available audio devices:")
            # for i, device in enumerate(self.devices):
            #     print(f"{i}: {device['name']} (inputs: {device['max_input_channels']})")

            # Select default input device
            self.device = None
            # Default device selection logic - can be simplified or made more robust
            # For a library class, direct device specification might be preferred over auto-selection.
            # Consider if sd.default.device is always reliable or if a specific device name/ID is better.
            # The original notebook code had a loop here to find the default.
            # For now, let's assume the user might need to set it explicitly if default isn't right.
            # We'll keep a simplified version of the device discovery for now.

            # Attempt to find a default input device (simplified)
            default_devices = sd.default.device
            if isinstance(default_devices, (list, tuple)) and len(default_devices) == 2: # Default input and output
                 default_input_device_index = default_devices[0]
                 if 0 <= default_input_device_index < len(self.devices) and self.devices[default_input_device_index]['max_input_channels'] > 0:
                     self.device = default_input_device_index
                    # print(f"Default input device selected: {self.devices[self.device]['name']}")

            if self.device is None: # If default not found or not suitable, pick first available
                for i, device_info in enumerate(self.devices):
                    if device_info['max_input_channels'] > 0:
                        self.device = i
                        # print(f"Selected first available input device: {self.devices[self.device]['name']}")
                        break
        except Exception as e:
            # print(f"Error querying devices: {e}") # UI related
            self.devices = [] # Ensure self.devices is initialized
            self.device = None # Ensure self.device is initialized

        # Initialize camera
        self.detect_camera() # Call unconditionally for now

    def detect_camera(self):
        # Try to find an available camera
        # print("Detecting cameras...") # UI related, remove for library code
        for i in range(5): # Check first 5 indices
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                # print(f"Camera found at index {i}") # UI related
                self.selected_camera_index = i
                cap.release() # Release it immediately
                return i
        # print("No camera detected.") # UI related
        self.selected_camera_index = -1
        return -1

    def _record_video_frames(self):
        if not self.video_capture or not self.video_writer:
            # print("Video capture or writer not initialized.") # Debug
            return

        # print("Starting video frame recording loop...") # Debug
        while self.recording and self.video_recording_enabled:
            ret, frame = self.video_capture.read()
            if ret:
                self.video_writer.write(frame)
            else:
                # print("Failed to read frame from video capture.") # Debug
                # Could add a small sleep here if ret is False to avoid tight loop on error
                time.sleep(0.01)
                # Or break if consistently failing, but for now, keep trying as long as recording is active
        # print("Video frame recording loop ended.") # Debug

    def set_device(self, device_id):
        """Set the recording device to use"""
        if not hasattr(self, 'devices') or not self.devices:
            # print("No devices available to set.") # UI related
            return False
        if 0 <= device_id < len(self.devices):
            if self.devices[device_id]['max_input_channels'] > 0:
                self.device = device_id
                # print(f"Selected device: {self.devices[device_id]['name']}") # UI related
                return True
            else:
                # print(f"Device {device_id} has no input channels") # UI related
                return False
        else:
            # print("Invalid device ID") # UI related
            return False

    def calibrate_noise(self):
        """Calibrate for background noise by listening for a few seconds"""
        # print("Calibrating for background noise... please ensure only regular background noise is present") # UI
        if self.device is None:
            # print("No recording device selected for calibration.") # UI
            return False

        try:
            recording = sd.rec(
                int(self.rate * self.calibration_time),
                samplerate=self.rate,
                channels=self.channels,
                device=self.device,
                blocking=True
            )
            sd.wait() # Wait for recording to complete

            if len(recording) > 0:
                rms = np.sqrt(np.mean(np.square(recording)))
                self.background_noise_level = rms
                # print(f"Background noise level: {rms:.6f}") # UI
                return True
            else:
                # print("No audio recorded during calibration") # UI
                return False

        except Exception as e:
            # print(f"Calibration error: {e}") # UI
            return False

    def audio_callback(self, indata, frames, time, status):
        """This function is called for each audio block"""
        if status:
            # print(status) # Can be logged
            pass
        self.frames.append(indata.copy())

        # Processing in callback should be minimal.
        # The original notebook had print statements here.
        # For a library, this might be better handled by a queue or event system
        # if real-time feedback is needed by the calling application.
        # For now, we remove the print from here.

    def start_recording(self):
        if self.recording:
            # print("Already recording!") # UI
            return False

        if self.device is None:
            # print("No recording device selected!") # UI
            return False

        if not self.calibrate_noise():
            # print("Calibration failed, using default threshold or no thresholding if not set.") # UI
            # Decide if recording should proceed without calibration, or return False
            pass # Proceeding without effective calibration for now

        # print("Starting sleep monitoring...") # UI
        self.recording = True
        self.frames = []

        try:
            self.start_time = datetime.datetime.now()
            # self.start_time_sec = time.time() # Not used in this version of audio_callback

            self.stream = sd.InputStream(
                samplerate=self.rate,
                channels=self.channels,
                device=self.device,
                callback=self.audio_callback
            )
            self.stream.start()

            self.save_thread = threading.Thread(target=self._save_recordings_periodically)
            self.save_thread.daemon = True
            self.save_thread.start()

            # Video recording start
            if self.video_recording_enabled and self.selected_camera_index != -1:
                try:
                    # Create session directory for video if it doesn't exist (audio part might have created it)
                    current_date_str = self.start_time.strftime("%Y-%m-%d")
                    session_dir = os.path.join(self.output_dir, current_date_str)
                    if not os.path.exists(session_dir):
                        os.makedirs(session_dir)

                    timestamp_str = self.start_time.strftime("%H-%M-%S")
                    self.video_filename = os.path.join(session_dir, f"sleep_video_{timestamp_str}.mp4")

                    self.video_capture = cv2.VideoCapture(self.selected_camera_index)
                    if not self.video_capture.isOpened():
                        raise Exception(f"Failed to open camera at index {self.selected_camera_index}")

                    frame_width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                    frame_height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = self.video_capture.get(cv2.CAP_PROP_FPS)
                    if fps == 0 or fps is None: # Handle cases where FPS is not reported or is zero
                        # print("Warning: Camera FPS reported as 0, defaulting to 20 FPS.") # UI/Log
                        fps = 20.0

                    # print(f"Video properties: {frame_width}x{frame_height} @ {fps} FPS") # Debug

                    self.video_writer = cv2.VideoWriter(
                        self.video_filename,
                        cv2.VideoWriter_fourcc(*'mp4v'),
                        fps,
                        (frame_width, frame_height)
                    )
                    if not self.video_writer.isOpened(): # isOpened for VideoWriter
                         raise Exception(f"Failed to open VideoWriter for {self.video_filename}")


                    self.video_thread = threading.Thread(target=self._record_video_frames)
                    self.video_thread.daemon = True
                    self.video_thread.start()
                    # print(f"Video recording started for camera index {self.selected_camera_index} to {self.video_filename}") # UI
                except Exception as ve:
                    # print(f"Error starting video recording: {ve}") # UI
                    self.video_recording_enabled = False # Disable if setup failed
                    if self.video_capture:
                        self.video_capture.release()
                        self.video_capture = None
                    if self.video_writer:
                        self.video_writer.release()
                        self.video_writer = None
                    self.video_filename = None

            # print("Sleep monitoring started.") # UI
            return True

        except Exception as e:
            self.recording = False
            # print(f"Error starting recording: {e}") # UI
            # Ensure video resources are cleaned up if audio part failed after video started
            if self.video_capture: self.video_capture.release()
            if self.video_writer: self.video_writer.release()
            return False

    def stop_recording(self):
        if not self.recording:
            # print("Not currently recording!") # UI
            return

        self.recording = False # Signal all recording threads to stop

        try:
            # Stop audio stream and save thread
            if hasattr(self, 'stream') and self.stream:
                self.stream.stop()
                self.stream.close()
            if hasattr(self, 'save_thread') and self.save_thread.is_alive():
                self.save_thread.join(timeout=5)
            self._final_save() # Save any remaining audio frames

            # Stop video thread and release resources
            if hasattr(self, 'video_thread') and self.video_thread.is_alive():
                self.video_thread.join(timeout=5)

            if self.video_capture:
                self.video_capture.release()
                self.video_capture = None
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None

            if self.video_filename: # Only print if we actually started video
                # print(f"Video recording stopped. Saved to {self.video_filename}") # UI
                self.video_filename = None

            # print("Sleep monitoring stopped.") # UI

        except Exception as e:
            # print(f"Error stopping recording: {e}") # UI
            # Ensure resources are attempted to be released even on error
            if hasattr(self, 'stream') and self.stream and self.stream.active: self.stream.stop(); self.stream.close()
            if self.video_capture: self.video_capture.release(); self.video_capture = None
            if self.video_writer: self.video_writer.release(); self.video_writer = None
            pass

    def _save_recordings_periodically(self):
        """Thread function to periodically save audio to files"""
        segment_duration = 30 * 60  # 30 minutes in seconds, make configurable?
        last_save_time = time.time()

        while self.recording:
            current_time = time.time()
            if current_time - last_save_time >= segment_duration:
                if self.frames: # Only save if there's something to save
                    self._save_segment()
                last_save_time = current_time
            time.sleep(1) # Check every second

    def _save_segment(self, is_final_save=False):
        """Save current audio frames to a file and analyze it."""
        if not self.frames:
            return

        # Create a local copy of frames to process and clear instance self.frames
        frames_to_save = self.frames[:]
        if not is_final_save: # If it's a periodic save, clear for next segment
             self.frames = []

        current_date = self.start_time.strftime("%Y-%m-%d") if self.start_time else datetime.datetime.now().strftime("%Y-%m-%d")
        session_dir = os.path.join(self.output_dir, current_date)
        if not os.path.exists(session_dir):
            os.makedirs(session_dir)

        timestamp = datetime.datetime.now().strftime("%H-%M-%S")
        filename_base = f"sleep_{timestamp}"
        filename_wav = os.path.join(session_dir, f"{filename_base}.wav")

        try:
            audio_data = np.concatenate(frames_to_save, axis=0)
            wav.write(filename_wav, self.rate, audio_data)
            # print(f"Saved recording to {filename_wav}") # UI / Logging

            self._analyze_segment(filename_wav, audio_data, filename_base, session_dir)

        except Exception as e:
            # print(f"Error saving audio segment: {e}") # UI / Logging
            pass

    def _final_save(self):
        """Save any remaining frames when stopping recording"""
        if self.frames:
            self._save_segment(is_final_save=True)
        self.frames = [] # Ensure frames are cleared after final save

    def _analyze_segment(self, wav_filename, audio_data, base_filename, session_dir):
        """Analyze the audio segment for interesting sleep events and save analysis."""
        try:
            analysis_txt_filename = os.path.join(session_dir, f"{base_filename}_analysis.txt")
            analysis_png_filename = os.path.join(session_dir, f"{base_filename}_analysis.png")

            sound_events = []
            chunk_size = int(self.rate)  # 1-second chunks

            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i:i+chunk_size]
                if len(chunk) > 0:
                    rms = np.sqrt(np.mean(np.square(chunk)))
                    if self.background_noise_level and rms > self.background_noise_level * 1.5: # Threshold is 1.5x background
                        time_point = i / self.rate
                        sound_events.append((time_point, rms))

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

            ax1.plot(np.arange(len(audio_data)) / self.rate, audio_data)
            ax1.set_title('Waveform')
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('Amplitude')

            f_spec, t_spec, Sxx = signal.spectrogram(audio_data.flatten(), self.rate)
            ax2.pcolormesh(t_spec, f_spec, 10 * np.log10(Sxx + 1e-9), shading='gouraud') # Added 1e-9 to avoid log10(0)
            ax2.set_title('Spectrogram')
            ax2.set_ylabel('Frequency (Hz)')
            ax2.set_xlabel('Time (s)')

            plt.tight_layout()
            plt.savefig(analysis_png_filename)
            plt.close(fig) # Close the figure to free memory

            with open(analysis_txt_filename, 'w') as f:
                f.write(f"Sleep Sound Analysis for {wav_filename}\n")
                f.write(f"Recorded at: {self.start_time.strftime('%Y-%m-%d %H:%M:%S') if self.start_time else 'N/A'}\n\n")
                duration = len(audio_data) / self.rate
                f.write(f"Total recording duration: {duration:.2f} seconds\n")
                if self.background_noise_level:
                    f.write(f"Background noise level (RMS): {self.background_noise_level:.6f}\n")

                if sound_events:
                    f.write(f"\nDetected {len(sound_events)} significant sound events (threshold: >1.5x background noise):\n")
                    for i, (time_point, level) in enumerate(sound_events):
                        f.write(f"  Event {i+1}: at {time_point:.2f}s (RMS level: {level:.6f})\n")
                else:
                    f.write("\nNo significant sound events detected above the threshold.\n")

            # print(f"Analysis saved to {analysis_txt_filename} and {analysis_png_filename}") # UI / Logging

        except Exception as e:
            # print(f"Error analyzing audio segment: {e}") # UI / Logging
            pass
