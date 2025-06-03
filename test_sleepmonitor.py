import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import os
import datetime
import shutil
import sys # Import sys for sys.modules patching

# Mock sounddevice first, before SleepMonitor is imported
mock_sd = MagicMock()
_mock_input_device = {'name': 'Mock Input Device', 'max_input_channels': 1, 'default_samplerate': 44100.0, 'hostapi': 0}
_mock_output_device = {'name': 'Mock Output Device', 'max_input_channels': 0, 'default_samplerate': 44100.0, 'hostapi': 0}
_mock_another_input_device = {'name': 'Another Mock Input', 'max_input_channels': 2, 'default_samplerate': 48000.0, 'hostapi': 0}
mock_sd.query_devices.return_value = [_mock_input_device, _mock_output_device, _mock_another_input_device]
mock_sd.default.device = [0, 1]
mock_sd.rec.return_value = np.array([[0.0]] * 16000)
mock_sd.InputStream = MagicMock()
mock_sd.wait.return_value = None

# Force sounddevice to be seen as the mock *before* sleepmonitor module is loaded
sys.modules['sounddevice'] = mock_sd

# --- Mock cv2 (OpenCV) ---
mock_cv2 = MagicMock()

# Mock cv2.VideoCapture class
mock_video_capture_instance = MagicMock()
mock_video_capture_instance.isOpened.return_value = True # Default for found camera
mock_video_capture_instance.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8)) # Dummy frame
mock_video_capture_instance.release = MagicMock()
def mock_video_capture_get(prop_id):
    if prop_id == mock_cv2.CAP_PROP_FRAME_WIDTH:
        return 640
    elif prop_id == mock_cv2.CAP_PROP_FRAME_HEIGHT:
        return 480
    elif prop_id == mock_cv2.CAP_PROP_FPS:
        return 30.0
    return 0
mock_video_capture_instance.get.side_effect = mock_video_capture_get

MockVideoCaptureClassFactory = MagicMock() # This will be the class VideoCapture
def mock_video_capture_constructor(index):
    # Reset relevant mocks for the instance each time a VideoCapture is "created"
    mock_video_capture_instance.isOpened.reset_mock()
    mock_video_capture_instance.read.reset_mock()
    mock_video_capture_instance.release.reset_mock()
    mock_video_capture_instance.get.reset_mock()

    if index == 0: # Simulate camera found at index 0
        mock_video_capture_instance.isOpened.return_value = True
    else:
        mock_video_capture_instance.isOpened.return_value = False
    return mock_video_capture_instance

MockVideoCaptureClassFactory.side_effect = mock_video_capture_constructor
mock_cv2.VideoCapture = MockVideoCaptureClassFactory

# Mock cv2.VideoWriter class
mock_video_writer_instance = MagicMock()
mock_video_writer_instance.write = MagicMock()
mock_video_writer_instance.release = MagicMock()
mock_video_writer_instance.isOpened.return_value = True # Simulate successful open

MockVideoWriterClassFactory = MagicMock(return_value=mock_video_writer_instance) # Class that returns the instance
mock_cv2.VideoWriter = MockVideoWriterClassFactory

mock_cv2.VideoWriter_fourcc = MagicMock(return_value=12345)
mock_cv2.CAP_PROP_FRAME_WIDTH = 3 # Using cv2 defined constants directly in get() mock
mock_cv2.CAP_PROP_FRAME_HEIGHT = 4
mock_cv2.CAP_PROP_FPS = 5


sys.modules['cv2'] = mock_cv2
# --- End Mock cv2 ---


# Now import SleepMonitor. It will import the mocked sounddevice and cv2.
from sleepmonitor import SleepMonitor

# The global mock_sd object is already configured.
# The @patch decorator on the class might be redundant if sys.modules works as expected,
# but it can be kept for belt-and-suspenders to ensure 'sleepmonitor.sd' is the mock.
# If sleepmonitor.py does "import sounddevice as sd", then sys.modules['sounddevice'] = mock_sd
# should ensure that 'sd' becomes 'mock_sd'.

@patch('sleepmonitor.sd', mock_sd) # Patch where 'sd' is used in sleepmonitor.py
class TestSleepMonitor(unittest.TestCase):

    def setUp(self):
        # Reset mocks that might hold state from other tests or previous runs if defined globally
        mock_video_capture_instance.reset_mock(return_value=True, side_effect=True)
        mock_video_writer_instance.reset_mock(return_value=True, side_effect=True)
        MockVideoCaptureClassFactory.reset_mock(return_value=True, side_effect=True)
        MockVideoWriterClassFactory.reset_mock(return_value=True, side_effect=True)

        # Re-configure side effects for constructors after reset
        MockVideoCaptureClassFactory.side_effect = mock_video_capture_constructor
        MockVideoWriterClassFactory.return_value = mock_video_writer_instance


        # Create a temporary directory for test recordings
        self.test_output_dir = "sleep_recordings_test_actual"
        if not os.path.exists(self.test_output_dir):
            os.makedirs(self.test_output_dir)
        # Now using the actual SleepMonitor class, it will use the mocked 'sd' and 'cv2'
        # Explicitly disable video for most tests by default. Specific tests can enable it.
        self.monitor = SleepMonitor(output_dir=self.test_output_dir, video_recording_enabled_by_default=False)
        # Check that camera detection was (mock) attempted.
        # If video_recording_enabled_by_default is False, selected_camera_index might still be set by detect_camera.
        self.assertEqual(self.monitor.selected_camera_index, 0)

    def tearDown(self):
        # Remove the temporary directory after tests
        if os.path.exists(self.test_output_dir):
            shutil.rmtree(self.test_output_dir)

    def _read_analysis_file(self, session_dir, base_filename_for_analysis):
        """Helper method to read the content of an analysis text file."""
        analysis_filename = os.path.join(session_dir, f"{base_filename_for_analysis}_analysis.txt")
        try:
            with open(analysis_filename, 'r') as f:
                return f.read()
        except FileNotFoundError:
            return None

    @patch('sleepmonitor.plt')
    @patch('sleepmonitor.signal')
    def test_analyze_segment_no_events(self, mock_signal, mock_plt):
        mock_signal.spectrogram.return_value = (np.array([1]), np.array([1]), np.array([[1]]))
        mock_plt.subplots.return_value = (MagicMock(), (MagicMock(), MagicMock()))
        mock_plt.tight_layout = MagicMock()
        mock_plt.savefig = MagicMock()
        mock_plt.close = MagicMock()

        self.monitor.background_noise_level = 0.1
        audio_data = np.random.rand(self.monitor.rate * 5) * 0.05
        self.monitor.start_time = datetime.datetime(2024, 1, 1, 10, 0, 0)
        base_wav_filename = "test_no_events.wav"
        wav_filepath = os.path.join(self.test_output_dir, base_wav_filename)
        session_dir = self.test_output_dir
        base_filename_for_analysis = base_wav_filename.replace('.wav', '')

        self.monitor._analyze_segment(wav_filepath, audio_data, base_filename_for_analysis, session_dir)

        analysis_content = self._read_analysis_file(session_dir, base_filename_for_analysis)
        self.assertIsNotNone(analysis_content)
        self.assertIn("No significant sound events detected", analysis_content)

    @patch('sleepmonitor.plt')
    @patch('sleepmonitor.signal')
    def test_analyze_segment_with_events(self, mock_signal, mock_plt):
        mock_signal.spectrogram.return_value = (np.array([1]), np.array([1]), np.array([[1]]))
        mock_plt.subplots.return_value = (MagicMock(), (MagicMock(), MagicMock()))
        mock_plt.tight_layout = MagicMock()
        mock_plt.savefig = MagicMock()
        mock_plt.close = MagicMock()

        self.monitor.background_noise_level = 0.1
        audio_data = np.full(self.monitor.rate * 5, 0.05)
        event_amplitude = 0.3
        event_start_sample = self.monitor.rate * 2
        event_end_sample = self.monitor.rate * 3
        audio_data[event_start_sample:event_end_sample] = event_amplitude

        self.monitor.start_time = datetime.datetime(2024, 1, 1, 10, 0, 0)
        base_wav_filename = "test_with_events.wav"
        wav_filepath = os.path.join(self.test_output_dir, base_wav_filename)
        session_dir = self.test_output_dir
        base_filename_for_analysis = base_wav_filename.replace('.wav', '')

        self.monitor._analyze_segment(wav_filepath, audio_data, base_filename_for_analysis, session_dir)

        analysis_content = self._read_analysis_file(session_dir, base_filename_for_analysis)
        self.assertIsNotNone(analysis_content)
        self.assertIn("Detected 1 significant sound events", analysis_content)
        self.assertIn(f"at 2.00s (RMS level: {event_amplitude:.6f})", analysis_content)

    @patch('sleepmonitor.plt')
    @patch('sleepmonitor.signal')
    def test_analyze_segment_empty_audio(self, mock_signal, mock_plt):
        mock_signal.spectrogram.return_value = (np.array([]), np.array([]), np.array([[]])) # Empty Sxx
        mock_plt.subplots.return_value = (MagicMock(), (MagicMock(), MagicMock()))
        mock_plt.tight_layout = MagicMock()
        mock_plt.savefig = MagicMock()
        mock_plt.close = MagicMock()

        self.monitor.background_noise_level = 0.1
        audio_data = np.array([]) # Empty audio data
        self.monitor.start_time = datetime.datetime(2024, 1, 1, 10, 0, 0)
        base_wav_filename = "test_empty_audio.wav"
        wav_filepath = os.path.join(self.test_output_dir, base_wav_filename)
        session_dir = self.test_output_dir
        base_filename_for_analysis = base_wav_filename.replace('.wav', '')

        self.monitor._analyze_segment(wav_filepath, audio_data, base_filename_for_analysis, session_dir)

        analysis_content = self._read_analysis_file(session_dir, base_filename_for_analysis)
        self.assertIsNotNone(analysis_content)
        # Expecting no events, and duration should be 0.00 seconds
        self.assertIn("No significant sound events detected", analysis_content)
        self.assertIn("Total recording duration: 0.00 seconds", analysis_content)

    @patch('sleepmonitor.plt')
    @patch('sleepmonitor.signal')
    def test_analyze_segment_no_background_noise_level_set(self, mock_signal, mock_plt):
        mock_signal.spectrogram.return_value = (np.array([1]), np.array([1]), np.array([[1]]))
        mock_plt.subplots.return_value = (MagicMock(), (MagicMock(), MagicMock()))
        mock_plt.tight_layout = MagicMock()
        mock_plt.savefig = MagicMock()
        mock_plt.close = MagicMock()

        self.monitor.background_noise_level = None # Explicitly None
        audio_data = np.full(self.monitor.rate * 5, 0.05)
        # Add a spike that would be an event if background_noise_level was set
        audio_data[self.monitor.rate*2 : self.monitor.rate*3] = 0.3

        self.monitor.start_time = datetime.datetime(2024, 1, 1, 10, 0, 0)
        base_wav_filename = "test_no_bgnl.wav"
        wav_filepath = os.path.join(self.test_output_dir, base_wav_filename)
        session_dir = self.test_output_dir
        base_filename_for_analysis = base_wav_filename.replace('.wav', '')

        self.monitor._analyze_segment(wav_filepath, audio_data, base_filename_for_analysis, session_dir)

        analysis_content = self._read_analysis_file(session_dir, base_filename_for_analysis)
        self.assertIsNotNone(analysis_content)
        # Even with a spike, no events should be detected if background_noise_level is None
        self.assertIn("No significant sound events detected", analysis_content)
        self.assertNotIn("Background noise level (RMS):", analysis_content) # Check it's not printed if None

    def test_initialization_default(self):
        # Test with actual class default output_dir
        monitor = SleepMonitor(output_dir="sleep_recordings")
        self.assertEqual(monitor.output_dir, "sleep_recordings")
        self.assertEqual(monitor.channels, 1)
        self.assertEqual(monitor.rate, 16000)
        self.assertEqual(monitor.chunk_seconds, 1)
        self.assertEqual(monitor.calibration_time, 5)
        self.assertIsNone(monitor.background_noise_level)
        self.assertFalse(monitor.recording)
        self.assertFalse(monitor.video_recording_enabled) # Check new default

    def test_initialization_custom(self):
        custom_dir = "custom_sleep_output_actual"
        if not os.path.exists(custom_dir):
            os.makedirs(custom_dir)

        monitor = SleepMonitor(
            output_dir=custom_dir,
            channels=2,
            rate=44100,
            chunk_seconds=2,
            calibration_time=10,
            video_recording_enabled_by_default=True # Test with video enabled
        )
        self.assertEqual(monitor.output_dir, custom_dir)
        self.assertEqual(monitor.channels, 2)
        self.assertEqual(monitor.rate, 44100)
        self.assertEqual(monitor.chunk_seconds, 2)
        self.assertEqual(monitor.calibration_time, 10)
        self.assertTrue(monitor.video_recording_enabled) # Check if True when set

        if os.path.exists(custom_dir):
            shutil.rmtree(custom_dir)

    @patch('sleepmonitor.threading.Thread') # Mock threading.Thread
    def test_start_stop_video_recording(self, MockThread):
        # Instantiate monitor with video enabled
        monitor = SleepMonitor(output_dir=self.test_output_dir, video_recording_enabled_by_default=True)
        self.assertTrue(monitor.video_recording_enabled)
        self.assertEqual(monitor.selected_camera_index, 0) # Mock camera detection should find index 0

        # Mock the target methods for threads to check if they are started
        mock_audio_save_thread_instance = MagicMock()
        mock_video_record_thread_instance = MagicMock()

        thread_map = {
            monitor._save_recordings_periodically: mock_audio_save_thread_instance,
            monitor._record_video_frames: mock_video_record_thread_instance
        }

        def mock_thread_constructor(target, daemon=False):
            if target in thread_map:
                # Return the specific mock instance for the known target
                thread_mock = thread_map[target]
                thread_mock.daemon = daemon # Set daemon attribute as the code does
                return thread_mock
            # Fallback for any other threads (should not happen in this test)
            fallback_mock = MagicMock()
            fallback_mock.daemon = daemon
            return fallback_mock

        MockThread.side_effect = mock_thread_constructor

        # Start recording
        monitor.start_recording()

        self.assertTrue(monitor.recording)
        self.assertIsNotNone(monitor.video_capture)
        self.assertIsNotNone(monitor.video_writer)

        # Check that VideoCapture methods were called
        mock_video_capture_instance.isOpened.assert_called()
        mock_video_capture_instance.get.assert_any_call(mock_cv2.CAP_PROP_FRAME_WIDTH)
        mock_video_capture_instance.get.assert_any_call(mock_cv2.CAP_PROP_FRAME_HEIGHT)
        mock_video_capture_instance.get.assert_any_call(mock_cv2.CAP_PROP_FPS)

        # Check that VideoWriter was instantiated and isOpened called on its instance
        MockVideoWriterClassFactory.assert_called_once() # Check constructor was called
        mock_video_writer_instance.isOpened.assert_called() # Check instance method

        # Check that the video recording thread was started
        mock_video_record_thread_instance.start.assert_called_once()

        # Simulate some time passing and frames being read (optional, more for integration)
        # For this test, primarily checking setup and teardown calls.

        # Stop recording
        monitor.stop_recording()

        self.assertFalse(monitor.recording)
        mock_video_capture_instance.release.assert_called_once()
        mock_video_writer_instance.release.assert_called_once()
        self.assertIsNone(monitor.video_capture)
        self.assertIsNone(monitor.video_writer)
        self.assertIsNone(monitor.video_filename)

        # Check that video thread was joined
        mock_video_record_thread_instance.join.assert_called_with(timeout=5)


if __name__ == '__main__':
    unittest.main()
