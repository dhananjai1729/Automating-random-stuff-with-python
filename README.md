# Sleep Monitor

This repository contains a Python script for sleep monitoring. The script is implemented as a Jupyter Notebook (`sleepmonitor.ipynb`).

## Features

- Audio recording using the microphone.
- Sound event detection based on background noise calibration.
- Automatic saving of recordings in WAV format.
- Generation of spectrograms and text-based analysis for each recording segment.
- Optional video recording using a connected webcam, synchronized with audio (saved as MP4).
- Recordings and analyses are stored in the `sleep_recordings` directory, organized by date.

## Technologies Used

- Python
- Jupyter Notebook
- sounddevice: For audio recording.
- opencv-python: For video capture.
- numpy: For numerical operations.
- scipy: For signal processing (spectrogram generation).
- matplotlib: For plotting spectrograms.
- ipywidgets: For creating interactive UI elements in the Jupyter Notebook.

## How to Use

1.  **Install dependencies:**
    Ensure you have Python 3.7+ installed. Then, install the required packages using the `requirements.txt` file. It's recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```
2.  **Run Jupyter Notebook:**
    ```bash
    jupyter notebook sleepmonitor.ipynb
    ```
3.  Open `sleepmonitor.ipynb` in your browser if it doesn't open automatically.
4.  Select the desired audio input device from the dropdown menu.
5.  (Optional) If you have a webcam and want to record video, check the "Enable Video Recording" checkbox. The system will attempt to use the first detected camera.
6.  Click "Start Recording" to begin monitoring. The script will calibrate for background noise (audio).
7.  Click "Stop Recording" to end the session.
8.  Recorded audio files (WAV), video files (MP4, if enabled and successful), spectrogram images (PNG), and analysis text files (TXT) will be saved in the `sleep_recordings` directory.

## Manual Testing for Video Recording

To manually test the video recording functionality:

1.  **Environment Setup**:
    *   Ensure you have a working webcam connected to your computer.
    *   Install all necessary dependencies, including `opencv-python`, by running:
        ```bash
        pip install -r requirements.txt
        ```
    *   Launch the Jupyter Notebook:
        ```bash
        jupyter notebook sleepmonitor.ipynb
        ```

2.  **Configure Recording**:
    *   In the notebook, select your desired audio input device.
    *   Check the "Enable Video Recording" checkbox.
    *   Observe the camera status output. It should indicate if a camera was detected by `SleepMonitor`.

3.  **Start Recording**:
    *   Click the "Start Recording" button.
    *   Verify that the status messages indicate both audio and video recording have started (if video was enabled and a camera detected).
    *   If a camera is physically present and active, you might see its indicator light turn on.

4.  **Perform Actions (Optional)**:
    *   While recording, make some movements or changes in the camera's field of view if you want to verify the content later.

5.  **Stop Recording**:
    *   Click the "Stop Recording" button after a short period (e.g., 30 seconds to 1 minute).
    *   Verify that the status messages indicate recording has stopped.

6.  **Check Output Files**:
    *   Navigate to the `sleep_recordings/` directory.
    *   Open the subdirectory corresponding to the current date.
    *   You should find:
        *   An audio file (e.g., `sleep_{timestamp}.wav`).
        *   If video recording was enabled and successful, a video file (e.g., `sleep_video_{timestamp}.mp4`).
        *   Analysis files for the audio.
    *   The timestamps for the audio and video files should be very close.

7.  **Verify Video Playback**:
    *   Open the generated MP4 video file using a media player.
    *   Confirm that the video plays correctly and shows the content captured by your webcam during the recording period.
    *   Check that the video duration is roughly what you expected.

8.  **Test Without Camera (Optional)**:
    *   If you want to test the behavior when no camera is available, try running the recording with video enabled but your webcam physically disconnected or disabled.
    *   The system should ideally handle this gracefully (e.g., record audio only, log a message that video could not be started). Check the `SleepMonitor`'s behavior in this case.