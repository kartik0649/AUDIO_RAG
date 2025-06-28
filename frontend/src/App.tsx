import React, { useState, useEffect } from 'react';
import { StyleSheet, Text, View, TouchableOpacity, ScrollView, ActivityIndicator, Alert, Animated } from 'react-native';
import { Audio, InterruptionModeIOS, InterruptionModeAndroid } from 'expo-av';
import * as FileSystem from 'expo-file-system';
import { Ionicons } from '@expo/vector-icons';

export default function App() {
  // Configure your backend URL here
  const BACKEND_URL = 'http://192.168.1.192:8001'; // Change this to your computer's IP address
  
  const [recording, setRecording] = useState<Audio.Recording | null>(null);
  const [transcript, setTranscript] = useState('');
  const [response, setResponse] = useState('');
  const [latency, setLatency] = useState(0);
  const [isLoading, setIsLoading] = useState(false);
  const [pulseAnim] = useState(new Animated.Value(1));
  const [buttonScale] = useState(new Animated.Value(1));
  const [isConnected, setIsConnected] = useState(false);

  // Check connection status
  useEffect(() => {
    const checkConnection = async () => {
      try {
        const response = await fetch(`${BACKEND_URL}/health`, {
          method: 'GET',
        });
        setIsConnected(response.ok);
      } catch (error) {
        setIsConnected(false);
      }
    };
    
    checkConnection();
    const interval = setInterval(checkConnection, 10000); // Check every 10 seconds
    return () => clearInterval(interval);
  }, [BACKEND_URL]);

  // Pulsing animation for recording indicator
  useEffect(() => {
    if (recording) {
      const pulse = Animated.loop(
        Animated.sequence([
          Animated.timing(pulseAnim, {
            toValue: 1.3,
            duration: 800,
            useNativeDriver: true,
          }),
          Animated.timing(pulseAnim, {
            toValue: 1,
            duration: 800,
            useNativeDriver: true,
          }),
        ])
      );
      pulse.start();
      return () => pulse.stop();
    } else {
      pulseAnim.setValue(1);
    }
  }, [recording, pulseAnim]);

  const animatePress = () => {
    Animated.sequence([
      Animated.timing(buttonScale, {
        toValue: 0.95,
        duration: 100,
        useNativeDriver: true,
      }),
      Animated.timing(buttonScale, {
        toValue: 1,
        duration: 100,
        useNativeDriver: true,
      }),
    ]).start();
  };

  const startRecording = async () => {
    try {
      // Request permissions with better error handling
      const permission = await Audio.requestPermissionsAsync();
      if (permission.status !== 'granted') {
        Alert.alert('Permission Required', 'Microphone permission is required to record audio.');
        return;
      }

      // Set audio mode with more compatible settings
      await Audio.setAudioModeAsync({
        allowsRecordingIOS: true,
        playsInSilentModeIOS: true,
        staysActiveInBackground: false,
        interruptionModeIOS: InterruptionModeIOS.DoNotMix,
        interruptionModeAndroid: InterruptionModeAndroid.DoNotMix,
        shouldDuckAndroid: true,
        playThroughEarpieceAndroid: false,
      });

      // Create and configure recording with simpler, more compatible options
      const rec = new Audio.Recording();
      
      // Try with minimal configuration first
      try {
        await rec.prepareToRecordAsync({
          android: {
            extension: '.wav',
            outputFormat: Audio.RecordingOptionsPresets.HIGH_QUALITY.android.outputFormat,
            audioEncoder: Audio.RecordingOptionsPresets.HIGH_QUALITY.android.audioEncoder,
            sampleRate: 16000,
            numberOfChannels: 1,
            bitRate: 128000,
          },
          ios: {
            extension: '.wav',
            outputFormat: Audio.RecordingOptionsPresets.HIGH_QUALITY.ios.outputFormat,
            audioQuality: Audio.RecordingOptionsPresets.HIGH_QUALITY.ios.audioQuality,
            sampleRate: 16000,
            numberOfChannels: 1,
            bitRate: 128000,
          },
          web: {
            mimeType: 'audio/wav',
            bitsPerSecond: 128000,
          },
        });
      } catch (prepareError: any) {
        console.log('First prepare attempt failed, trying with default settings:', prepareError);
        // Fallback to default settings
        await rec.prepareToRecordAsync(Audio.RecordingOptionsPresets.HIGH_QUALITY);
      }

      await rec.startAsync();
      setRecording(rec);
      console.log('Recording started successfully');
    } catch (err: any) {
      console.error('Failed to start recording', err);
      const errorMessage = err?.message || err?.toString() || 'Unknown error occurred';
      Alert.alert('Recording Error', `Failed to start recording: ${errorMessage}`);
    }
  };

  const stopRecording = async () => {
    if (!recording) return;
    
    setIsLoading(true);
    try {
      await recording.stopAndUnloadAsync();
      const uri = recording.getURI();
      setRecording(null);
      
      if (!uri) {
        setIsLoading(false);
        Alert.alert('Error', 'No recording URI available');
        return;
      }

      console.log('Recording stopped, processing audio...');
      const data = await FileSystem.readAsStringAsync(uri, { encoding: FileSystem.EncodingType.Base64 });
      const start = Date.now();
      
      // Convert base64 to binary data for React Native
      const binaryString = atob(data);
      const bytes = new Uint8Array(binaryString.length);
      for (let i = 0; i < binaryString.length; i++) {
        bytes[i] = binaryString.charCodeAt(i);
      }
      
      console.log(`Sending ${bytes.length} bytes to backend...`);
      const res = await fetch(`${BACKEND_URL}/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/octet-stream' },
        body: bytes,
      });
      
      if (!res.ok) {
        throw new Error(`Network response was not ok: ${res.status}`);
      }
      
      const json = await res.json();
      setLatency(Date.now() - start);
      setTranscript(json.transcript);
      setResponse(json.response);
      console.log('Audio processed successfully');
    } catch (err: any) {
      console.error('Failed to process audio', err);
      const errorMessage = err?.message || err?.toString() || 'Unknown error occurred';
      Alert.alert('Processing Error', `Failed to process audio: ${errorMessage}`);
    } finally {
      setIsLoading(false);
    }
  };

  const clearChat = () => {
    setTranscript('');
    setResponse('');
    setLatency(0);
  };

  return (
    <View style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <View style={styles.headerContent}>
          <Ionicons name="mic" size={32} color="#6366f1" />
          <Text style={styles.headerTitle}>Voice Agent</Text>
          <Text style={styles.headerSubtitle}>AI-Powered Voice Assistant</Text>
          <View style={styles.connectionStatus}>
            <View style={[styles.statusDot, { backgroundColor: isConnected ? '#10b981' : '#ef4444' }]} />
            <Text style={[styles.statusText, { color: isConnected ? '#10b981' : '#ef4444' }]}>
              {isConnected ? 'Connected' : 'Disconnected'}
            </Text>
          </View>
        </View>
      </View>

      {/* Main Content */}
      <ScrollView style={styles.content} showsVerticalScrollIndicator={false}>
        {/* Recording Status */}
        <View style={styles.statusContainer}>
          {recording && (
            <View style={styles.recordingIndicator}>
              <Animated.View 
                style={[
                  styles.pulseDot,
                  { transform: [{ scale: pulseAnim }] }
                ]} 
              />
              <Text style={styles.recordingText}>Recording...</Text>
            </View>
          )}
          
          {isLoading && (
            <View style={styles.loadingContainer}>
              <ActivityIndicator size="large" color="#6366f1" />
              <Text style={styles.loadingText}>Processing your voice...</Text>
            </View>
          )}
        </View>

        {/* Chat Messages */}
        {(transcript || response) && (
          <View style={styles.chatContainer}>
            {transcript && (
              <View style={styles.messageContainer}>
                <View style={styles.userMessage}>
                  <Ionicons name="person" size={20} color="#6366f1" style={styles.messageIcon} />
                  <Text style={styles.userMessageText}>{transcript}</Text>
                </View>
              </View>
            )}
            
            {response && (
              <View style={styles.messageContainer}>
                <View style={styles.assistantMessage}>
                  <Ionicons name="chatbubble" size={20} color="#10b981" style={styles.messageIcon} />
                  <Text style={styles.assistantMessageText}>{response}</Text>
                </View>
              </View>
            )}
            
            {latency > 0 && (
              <View style={styles.latencyContainer}>
                <Ionicons name="time" size={16} color="#6b7280" />
                <Text style={styles.latencyText}>Response time: {latency}ms</Text>
              </View>
            )}
          </View>
        )}

        {/* Empty State */}
        {!transcript && !response && !recording && !isLoading && (
          <View style={styles.emptyState}>
            <Ionicons 
              name={isConnected ? "mic-outline" : "wifi-outline"} 
              size={64} 
              color={isConnected ? "#d1d5db" : "#ef4444"} 
            />
            <Text style={styles.emptyStateTitle}>
              {isConnected ? 'Ready to Chat' : 'Connection Required'}
            </Text>
            <Text style={styles.emptyStateSubtitle}>
              {isConnected 
                ? 'Tap the microphone button below to start a voice conversation'
                : `Please start the backend server on ${BACKEND_URL.replace('http://', '')} to use the voice agent`
              }
            </Text>
          </View>
        )}
      </ScrollView>

      {/* Bottom Controls */}
      <View style={styles.bottomContainer}>
        <View style={styles.controlsContainer}>
          {/* Clear Button */}
          {(transcript || response) && (
            <TouchableOpacity style={styles.clearButton} onPress={clearChat}>
              <Ionicons name="trash-outline" size={20} color="#ef4444" />
              <Text style={styles.clearButtonText}>Clear</Text>
            </TouchableOpacity>
          )}
          
          {/* Record Button */}
          <TouchableOpacity
            style={[
              styles.recordButton,
              recording && styles.recordButtonActive,
              (isLoading || !isConnected) && styles.recordButtonDisabled
            ]}
            onPress={() => {
              if (!isConnected) {
                Alert.alert('Connection Error', `Please make sure the backend server is running on ${BACKEND_URL.replace('http://', '')}`);
                return;
              }
              animatePress();
              if (recording) {
                stopRecording();
              } else {
                startRecording();
              }
            }}
            disabled={isLoading || !isConnected}
          >
            <Animated.View style={[{ transform: [{ scale: buttonScale }] }]}>
              <Ionicons
                name={recording ? "stop" : "mic"}
                size={32}
                color={recording ? "#ffffff" : "#6366f1"}
              />
            </Animated.View>
          </TouchableOpacity>
        </View>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f8fafc',
  },
  header: {
    backgroundColor: '#ffffff',
    paddingTop: 60,
    paddingBottom: 20,
    paddingHorizontal: 20,
    borderBottomWidth: 1,
    borderBottomColor: '#e2e8f0',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  headerContent: {
    alignItems: 'center',
  },
  headerTitle: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#1f2937',
    marginTop: 8,
  },
  headerSubtitle: {
    fontSize: 16,
    color: '#6b7280',
    marginTop: 4,
  },
  content: {
    flex: 1,
    paddingHorizontal: 20,
  },
  statusContainer: {
    alignItems: 'center',
    marginVertical: 20,
  },
  recordingIndicator: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#fef3c7',
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 20,
    borderWidth: 1,
    borderColor: '#f59e0b',
  },
  pulseDot: {
    width: 12,
    height: 12,
    borderRadius: 6,
    backgroundColor: '#ef4444',
    marginRight: 8,
  },
  recordingText: {
    color: '#92400e',
    fontWeight: '600',
  },
  loadingContainer: {
    alignItems: 'center',
  },
  loadingText: {
    marginTop: 12,
    color: '#6b7280',
    fontSize: 16,
  },
  chatContainer: {
    marginBottom: 20,
  },
  messageContainer: {
    marginBottom: 16,
  },
  userMessage: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    backgroundColor: '#6366f1',
    padding: 16,
    borderRadius: 20,
    borderBottomLeftRadius: 4,
    maxWidth: '85%',
    alignSelf: 'flex-end',
  },
  assistantMessage: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    backgroundColor: '#ffffff',
    padding: 16,
    borderRadius: 20,
    borderBottomRightRadius: 4,
    maxWidth: '85%',
    alignSelf: 'flex-start',
    borderWidth: 1,
    borderColor: '#e2e8f0',
  },
  messageIcon: {
    marginRight: 8,
    marginTop: 2,
  },
  userMessageText: {
    color: '#ffffff',
    fontSize: 16,
    lineHeight: 22,
    flex: 1,
  },
  assistantMessageText: {
    color: '#1f2937',
    fontSize: 16,
    lineHeight: 22,
    flex: 1,
  },
  latencyContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    marginTop: 8,
  },
  latencyText: {
    color: '#6b7280',
    fontSize: 14,
    marginLeft: 4,
  },
  emptyState: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    paddingVertical: 60,
  },
  emptyStateTitle: {
    fontSize: 24,
    fontWeight: '600',
    color: '#374151',
    marginTop: 16,
    marginBottom: 8,
  },
  emptyStateSubtitle: {
    fontSize: 16,
    color: '#6b7280',
    textAlign: 'center',
    paddingHorizontal: 40,
    lineHeight: 24,
  },
  bottomContainer: {
    backgroundColor: '#ffffff',
    paddingVertical: 20,
    paddingHorizontal: 20,
    borderTopWidth: 1,
    borderTopColor: '#e2e8f0',
  },
  controlsContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
  },
  recordButton: {
    width: 80,
    height: 80,
    borderRadius: 40,
    backgroundColor: '#ffffff',
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 3,
    borderColor: '#6366f1',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.15,
    shadowRadius: 8,
    elevation: 8,
  },
  recordButtonActive: {
    backgroundColor: '#ef4444',
    borderColor: '#ef4444',
  },
  recordButtonDisabled: {
    opacity: 0.6,
  },
  clearButton: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 20,
    backgroundColor: '#fef2f2',
    borderWidth: 1,
    borderColor: '#fecaca',
    marginRight: 20,
  },
  clearButtonText: {
    color: '#ef4444',
    fontWeight: '600',
    marginLeft: 4,
  },
  connectionStatus: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: 8,
  },
  statusDot: {
    width: 12,
    height: 12,
    borderRadius: 6,
    marginRight: 8,
  },
  statusText: {
    fontSize: 16,
    fontWeight: '600',
  },
});
