import unittest
from unittest.mock import patch, MagicMock, mock_open
import numpy as np
import os
import tempfile
import time
import datetime
import threading

# Import the SleepMonitor class (assuming it's in a module called sleep_monitor)
# For testing purposes, we'll create a mock version to test
from unittest.mock import Mock

class TestSleepMonitor(unittest.TestCase):
    
    def setUp(self):
        # Create a temporary directory for test outputs
        self.test_output_dir = tempfile.mkdtemp()
        
        # Mock the sounddevice module
        self.sd_patcher = patch('sounddevice.InputStream')
        self.mock_sd = self.sd_patcher.start()
        
        # Mock device query
        self.query_devices_patcher = patch('sounddevice.query_devices')
        self.mock_query_devices = self.query_devices_patcher.start()
        
        # Set up mock devices
        self.mock_devices = [
            {'name': 'Test Microphone', 'max_input_channels': 2},
            {'name': 'Test Speakers', 'max_input_channels': 0}
        ]
        self.mock_query_devices.return_value = self.mock_devices
        
        # Set up mock default device
        self.default_device_patcher = patch('sounddevice.default')
        self.mock_default_device = self.default_device_patcher.start()
        self.mock_default_device.device = [0, 1]  # Input, Output
        
        # Create a SleepMonitor instance with mocked dependencies
        with patch('os.makedirs'):
            self.monitor = SleepMonitor(output_dir=self.test_output_dir)
    
    def tearDown(self):
        # Clean up patches
        self.sd_patcher.stop()
        self.query_devices_patcher.stop()
        self.default_device_patcher.stop()
        
        # Clean up temporary directory
        import shutil
        shutil.rmtree(self.test_output_dir, ignore_errors=True)
        
    def test_initialization(self):
        """Test that SleepMonitor initializes correctly"""
        self.assertEqual(self.monitor.output_dir, self.test_output_dir)
        self.assertEqual(self.monitor.channels, 1)
        self.assertEqual(self.monitor.rate, 16000)
        self.assertEqual(self.monitor.device, 0)  # Should select first input device
        self.assertFalse(self.monitor.recording)
        self.assertIsNone(self.monitor.background_noise_level)
    
    @patch('os.makedirs')
    def test_set_device(self, mock_makedirs):
        """Test device selection functionality"""
        # Test setting valid device
        self.monitor.set_device(0)
        self.assertEqual(self.monitor.device, 0)
        
        # Test setting invalid device (no input channels)
        with patch('builtins.print') as mock_print:
            self.monitor.set_device(1)
            mock_print.assert_called_with("Device 1 has no input channels")
        
        # Test setting out of range device
        with patch('builtins.print') as mock_print:
            self.monitor.set_device(99)
            mock_print.assert_called_with("Invalid device ID")
    
    @patch('sounddevice.rec')
    def test_calibrate_noise(self, mock_rec):
        """Test background noise calibration"""
        # Create mock recording data
        mock_recording = np.array([0.1, 0.2, -0.1, 0.05]).reshape(-1, 1)
        mock_rec.return_value = mock_recording
        
        # Call calibrate_noise
        result = self.monitor.calibrate_noise()
        
        # Check results
        self.assertTrue(result)
        self.assertIsNotNone(self.monitor.background_noise_level)
        expected_rms = np.sqrt(np.mean(np.square(mock_recording)))
        self.assertAlmostEqual(self.monitor.background_noise_level, expected_rms)
        
        # Test error handling
        mock_rec.side_effect = Exception("Test error")
        with patch('builtins.print') as mock_print:
            result = self.monitor.calibrate_noise()
            self.assertFalse(result)
            mock_print.assert_called_with("Calibration error: Test error")
    
    def test_audio_callback(self):
        """Test the audio callback function"""
        # Initialize monitor
        self.monitor.frames = []
        self.monitor.start_time_sec = time.time()
        self.monitor.background_noise_level = 0.1
        
        # Create mock time and status
        mock_time = MagicMock()
        mock_time.currentTime = time.time()
        mock_status = None
        
        # Create test audio data - below threshold
        quiet_data = np.array([0.05, 0.06, -0.04, 0.03]).reshape(-1, 1)
        with patch('builtins.print') as mock_print:
            self.monitor.audio_callback(quiet_data, len(quiet_data), mock_time, mock_status)
            # Should append data but not print event
            self.assertEqual(len(self.monitor.frames), 1)
            mock_print.assert_not_called()
        
        # Test audio data above threshold
        loud_data = np.array([0.2, 0.3, -0.25, 0.4]).reshape(-1, 1)
        with patch('builtins.print') as mock_print:
            with patch('datetime.datetime') as mock_datetime:
                mock_datetime.now.return_value = datetime.datetime(2023, 1, 1, 12, 0, 0)
                self.monitor.audio_callback(loud_data, len(loud_data), mock_time, mock_status)
                # Should append data and print event
                self.assertEqual(len(self.monitor.frames), 2)
                mock_print.assert_called()  # Sound event detected
    
    def test_start_recording(self):
        """Test starting the recording process"""
        # Mock calibrate_noise to avoid actual recording
        with patch.object(self.monitor, 'calibrate_noise', return_value=True):
            # Mock stream methods
            mock_stream = MagicMock()
            self.mock_sd.return_value = mock_stream
            
            # Test starting recording
            with patch('threading.Thread') as mock_thread:
                mock_thread_instance = MagicMock()
                mock_thread.return_value = mock_thread_instance
                
                self.monitor.start_recording()
                
                # Check that recording started correctly
                self.assertTrue(self.monitor.recording)
                self.assertIsNotNone(self.monitor.start_time)
                self.assertIsNotNone(self.monitor.start_time_sec)
                mock_stream.start.assert_called_once()
                mock_thread.assert_called_once()
                mock_thread_instance.start.assert_called_once()
    
    def test_stop_recording(self):
        """Test stopping the recording process"""
        # Set up mocks
        self.monitor.recording = True
        self.monitor.stream = MagicMock()
        self.monitor.save_thread = MagicMock()
        
        # Mock _final_save to avoid actual saving
        with patch.object(self.monitor, '_final_save') as mock_final_save:
            self.monitor.stop_recording()
            
            # Check that recording stopped correctly
            self.assertFalse(self.monitor.recording)
            self.monitor.stream.stop.assert_called_once()
            self.monitor.stream.close.assert_called_once()
            self.monitor.save_thread.join.assert_called_once()
            mock_final_save.assert_called_once()
    
    def test_save_current_segment(self):
        """Test saving audio segments"""
        # Create test frames
        self.monitor.frames = [np.array([0.1, 0.2]).reshape(-1, 1), np.array([0.3, 0.4]).reshape(-1, 1)]
        self.monitor.rate = 16000
        
        # Mock directory creation and wav writing
        with patch('os.makedirs') as mock_makedirs:
            with patch('scipy.io.wavfile.write') as mock_wav_write:
                with patch.object(self.monitor, '_analyze_segment') as mock_analyze:
                    with patch('datetime.datetime') as mock_datetime:
                        # Set fixed datetime for predictable filenames
                        mock_datetime.now.return_value = datetime.datetime(2023, 1, 1, 12, 0, 0)
                        
                        self.monitor._save_current_segment()
                        
                        # Check directory creation
                        expected_dir = os.path.join(self.test_output_dir, "2023-01-01")
                        mock_makedirs.assert_called_with(expected_dir)
                        
                        # Check wav file writing
                        expected_file = os.path.join(expected_dir, "sleep_12-00-00.wav")
                        mock_wav_write.assert_called_once()
                        self.assertEqual(mock_wav_write.call_args[0][0], expected_file)
                        
                        # Check audio data
                        actual_data = mock_wav_write.call_args[0][2]
                        expected_data = np.concatenate(self.monitor.frames, axis=0)
                        np.testing.assert_array_equal(actual_data, expected_data)
                        
                        # Ensure analysis was called
                        mock_analyze.assert_called_once()
                        
                        # Check frames were cleared
                        self.assertEqual(len(self.monitor.frames), 0)
    
    def test_analyze_segment(self):
        """Test audio analysis functionality"""
        # Create test audio data with one "event"
        audio_data = np.zeros((16000,1))  # 1 second of silence
        # Add a loud section
        audio_data[8000:8500] = 0.5  # Half second of loud audio
        
        self.monitor.rate = 16000
        self.monitor.background_noise_level = 0.1
        
        # Mock file operations
        with patch('builtins.open', mock_open()) as mock_file:
            with patch('matplotlib.pyplot.savefig') as mock_savefig:
                with patch('matplotlib.pyplot.close') as mock_close:
                    filename = os.path.join(self.test_output_dir, "test.wav")
                    self.monitor._analyze_segment(filename, audio_data)
                    
                    # Check that file operations occurred
                    mock_file.assert_called_with(filename.replace('.wav', '_analysis.txt'), 'w')
                    mock_savefig.assert_called_with(filename.replace('.wav', '_analysis.png'))
                    mock_close.assert_called_once()
                    
                    # Check that write operations occurred (file content)
                    handle = mock_file()
                    handle.write.assert_called()
                    # We can't easily check the exact content, but we can check that write was called

if __name__ == "__main__":
    unittest.main()
