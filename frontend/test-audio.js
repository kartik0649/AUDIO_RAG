// Simple test to check if expo-av audio recording works
import { Audio } from 'expo-av';

export const testAudioRecording = async () => {
  try {
    console.log('Testing audio recording...');
    
    // Request permissions
    const permission = await Audio.requestPermissionsAsync();
    console.log('Permission status:', permission.status);
    
    if (permission.status !== 'granted') {
      console.error('Permission not granted');
      return false;
    }
    
    // Set audio mode
    await Audio.setAudioModeAsync({
      allowsRecordingIOS: true,
      playsInSilentModeIOS: true,
    });
    console.log('Audio mode set successfully');
    
    // Create recording
    const recording = new Audio.Recording();
    console.log('Recording object created');
    
    // Try to prepare with default settings
    await recording.prepareToRecordAsync(Audio.RecordingOptionsPresets.HIGH_QUALITY);
    console.log('Recording prepared successfully');
    
    // Start recording
    await recording.startAsync();
    console.log('Recording started');
    
    // Stop after 2 seconds
    setTimeout(async () => {
      await recording.stopAndUnloadAsync();
      console.log('Recording stopped');
      const uri = recording.getURI();
      console.log('Recording URI:', uri);
    }, 2000);
    
    return true;
  } catch (error) {
    console.error('Audio recording test failed:', error);
    return false;
  }
}; 