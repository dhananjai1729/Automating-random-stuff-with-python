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

# Now import SleepMonitor. It will import the mocked sounddevice.
from sleepmonitor import SleepMonitor

# The global mock_sd object is already configured.
# The @patch decorator on the class might be redundant if sys.modules works as expected,
# but it can be kept for belt-and-suspenders to ensure 'sleepmonitor.sd' is the mock.
# If sleepmonitor.py does "import sounddevice as sd", then sys.modules['sounddevice'] = mock_sd
# should ensure that 'sd' becomes 'mock_sd'.

@patch('sleepmonitor.sd', mock_sd) # Patch where 'sd' is used in sleepmonitor.py
class TestSleepMonitor(unittest.TestCase):

    def setUp(self):
        # Create a temporary directory for test recordings
        self.test_output_dir = "sleep_recordings_test_actual"
        if not os.path.exists(self.test_output_dir):
            os.makedirs(self.test_output_dir)
        # Now using the actual SleepMonitor class, it will use the mocked 'sd'
        self.monitor = SleepMonitor(output_dir=self.test_output_dir)

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

    def test_initialization_custom(self):
        custom_dir = "custom_sleep_output_actual"
        if not os.path.exists(custom_dir):
            os.makedirs(custom_dir)

        monitor = SleepMonitor(output_dir=custom_dir, channels=2, rate=44100, chunk_seconds=2, calibration_time=10)
        self.assertEqual(monitor.output_dir, custom_dir)
        self.assertEqual(monitor.channels, 2)
        self.assertEqual(monitor.rate, 44100)
        self.assertEqual(monitor.chunk_seconds, 2)
        self.assertEqual(monitor.calibration_time, 10)

        if os.path.exists(custom_dir):
            shutil.rmtree(custom_dir)

if __name__ == '__main__':
    unittest.main()
