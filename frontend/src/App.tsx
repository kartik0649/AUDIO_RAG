import React, { useState, useEffect, useRef } from 'react';
import { StyleSheet, Text, View, TouchableOpacity, ScrollView, ActivityIndicator, Alert, Animated } from 'react-native';
import { Audio, InterruptionModeIOS, InterruptionModeAndroid } from 'expo-av';
import * as FileSystem from 'expo-file-system';
import { Ionicons } from '@expo/vector-icons';
import EventSource from 'react-native-sse';

export default function App() {
  // Configure your backend URL here
  const BACKEND_URL = 'http://192.168.1.192:8001'; // Change this to your computer's IP address
  
  const [recording, setRecording] = useState<Audio.Recording | null>(null);
  const [transcript, setTranscript] = useState('');
  const [response, setResponse] = useState('');
  const [latency, setLatency] = useState(0);
  const [latencyStats, setLatencyStats] = useState<{
    audio_processing_latency: number;
    retrieval_latency: number;
    llm_latency: number;
    total_latency: number;
  } | null>(null);
  const [startTime, setStartTime] = useState<number | null>(null);
  const [retrievedDocuments, setRetrievedDocuments] = useState<Array<{
    source: string;
    original_file: string;
    similarity_score: number;
    chunk_index?: number;
    total_chunks?: number;
    token_count?: number;
  }>>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [pulseAnim] = useState(new Animated.Value(1));
  const [buttonScale] = useState(new Animated.Value(1));
  const [isConnected, setIsConnected] = useState(false);
  const [streamingResponse, setStreamingResponse] = useState('');
  const [streamingTranscript, setStreamingTranscript] = useState('');
  const [streamingStatus, setStreamingStatus] = useState('');
  const recordingRef = useRef<Audio.Recording | null>(null);
  const eventSourceRef = useRef<EventSource | null>(null);

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
      console.log('Requesting permissions...');
      const permission = await Audio.requestPermissionsAsync();
      if (permission.status !== 'granted') {
        Alert.alert('Permission required', 'Please grant microphone permission to record audio.');
        return;
      }

      console.log('Setting audio mode...');
      await Audio.setAudioModeAsync({
        allowsRecordingIOS: true,
        playsInSilentModeIOS: true,
      });

      console.log('Starting recording...');
      const { recording } = await Audio.Recording.createAsync(
        Audio.RecordingOptionsPresets.HIGH_QUALITY
      );
      recordingRef.current = recording;
      setRecording(recording);
      setStartTime(Date.now()); // Start timing
      console.log('Recording started');
    } catch (err) {
      console.error('Failed to start recording', err);
      Alert.alert('Recording Error', 'Failed to start recording');
    }
  };

  const stopRecording = async () => {
    console.log('stopRecording called, recordingRef.current:', recordingRef.current);
    if (!recordingRef.current) {
      console.log('No recording ref, returning');
      return;
    }

    try {
      console.log('Stopping recording...');
      await recordingRef.current.stopAndUnloadAsync();
      const uri = recordingRef.current.getURI();
      console.log('Recording stopped, URI:', uri);
      
      setRecording(null);
      recordingRef.current = null;
      setIsLoading(true);
      setStreamingResponse('');
      setStreamingTranscript('');
      setStreamingStatus('');

      if (uri) {
        await processAudio(uri);
      } else {
        console.log('No URI returned from recording');
        setIsLoading(false);
      }
    } catch (err) {
      console.error('Failed to stop recording', err);
      Alert.alert('Recording Error', 'Failed to stop recording');
      setIsLoading(false);
    }
  };

  const processAudio = async (audioUri: string) => {
    setIsLoading(true);
    setResponse('');
    setTranscript('');
    setLatency(0);
    setLatencyStats(null);
    setRetrievedDocuments([]);
    setStreamingResponse('');
    setStreamingTranscript('');
    setStreamingStatus('');

    try {
      console.log('Reading audio file...');
      
      // For React Native, we need to read the file as base64 first
      const base64Data = await FileSystem.readAsStringAsync(audioUri, {
        encoding: FileSystem.EncodingType.Base64,
      });
      
      console.log('Audio file read, size:', base64Data.length);
      
      // Convert base64 to binary string
      const binaryString = atob(base64Data);
      const bytes = new Uint8Array(binaryString.length);
      for (let i = 0; i < binaryString.length; i++) {
        bytes[i] = binaryString.charCodeAt(i);
      }
      
      console.log('Audio bytes created, size:', bytes.length);

      console.log('Sending audio to hybrid endpoint...');
      
      // Send the audio data directly as raw bytes
      const hybridResponse = await fetch(`${BACKEND_URL}/query/hybrid`, {
        method: 'POST',
        headers: {
          'Content-Type': 'audio/m4a',
        },
        body: bytes,
      });

      if (!hybridResponse.ok) {
        const errorText = await hybridResponse.text();
        console.error('Hybrid endpoint error:', hybridResponse.status, errorText);
        throw new Error(`Hybrid endpoint failed: ${hybridResponse.status} - ${errorText}`);
      }

      const { session_id } = await hybridResponse.json();
      console.log('Session ID received:', session_id);

      // Step 2: Connect to SSE for streaming
      console.log('Connecting to SSE stream...');
      
      // Close any existing EventSource
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }

      // Create new EventSource for real-time streaming
      const eventSource = new EventSource(`${BACKEND_URL}/stream/${session_id}`);
      eventSourceRef.current = eventSource;

      let fullResponse = '';
      let fullTranscript = '';
      let isComplete = false;
      let responseComplete = false;

      eventSource.addEventListener('open', (event) => {
        console.log('SSE connection opened');
      });

      eventSource.addEventListener('message', (event) => {
        try {
          if (!event.data) {
            console.log('Empty SSE message received');
            return;
          }
          const data = JSON.parse(event.data);
          console.log('SSE message received:', data);

          switch (data.type) {
            case 'status':
              console.log('Status:', data.message);
              setStreamingStatus(data.message);
              break;

            case 'transcript':
              fullTranscript = data.text;
              setStreamingTranscript(data.text);
              setTranscript(data.text);
              console.log('🟢 Transcript received and set:', data.text);
              console.log('🟢 Current transcript state:', fullTranscript);
              break;

            case 'chunk':
              fullResponse += data.content;
              setStreamingResponse(fullResponse);
              setResponse(fullResponse);
              console.log('Chunk received:', data.content);
              break;

            case 'complete':
              isComplete = true;
              responseComplete = true;
              fullResponse = data.full_response;
              setResponse(fullResponse);
              setTranscript(fullTranscript);
              setLatencyStats({
                audio_processing_latency: data.audio_processing_latency || 0,
                retrieval_latency: data.retrieval_latency || 0,
                llm_latency: data.llm_latency || 0,
                total_latency: data.total_latency || 0
              });
              console.log('🟢 Response complete:', data);
              console.log('🟢 Retrieved documents:', data.retrieved_documents);
              
              // Calculate end-to-end latency from start to finish
              const endTime = Date.now();
              const endToEndLatency = startTime ? endTime - startTime : 0;
              setLatency(endToEndLatency);
              
              // Use backend latency stats for breakdown
              const totalBackendLatency = data.total_latency || 0;
              
              console.log('🟢 Latency calculations:');
              console.log('  - Start time:', startTime);
              console.log('  - End time:', endTime);
              console.log('  - End-to-end latency:', endToEndLatency, 'ms');
              console.log('  - Backend total latency:', totalBackendLatency * 1000, 'ms');
              console.log('  - Network + UI latency:', Math.max(0, endToEndLatency - (totalBackendLatency * 1000)), 'ms');
              
              // Add retrieved documents from backend
              if (data.retrieved_documents && Array.isArray(data.retrieved_documents)) {
                console.log('🟢 Setting retrieved documents:', data.retrieved_documents);
                setRetrievedDocuments(data.retrieved_documents);
              } else {
                console.log('🟡 No retrieved documents in response');
              }
              
              // Clean up
              eventSource.close();
              eventSourceRef.current = null;
              setIsLoading(false);
              setStreamingResponse('');
              setStreamingTranscript('');
              setStreamingStatus('Complete');
              break;

            case 'error':
              console.error('SSE error:', data.message);
              // Only show error if response is not complete
              if (!responseComplete) {
                Alert.alert('Streaming Error', data.message);
              }
              eventSource.close();
              eventSourceRef.current = null;
              setIsLoading(false);
              break;

            default:
              console.log('Unknown message type:', data.type);
          }
        } catch (error) {
          console.error('Error parsing SSE message:', error);
        }
      });

      eventSource.addEventListener('error', (error) => {
        console.error('EventSource error:', error);
        if (!isComplete && !responseComplete) {
          Alert.alert('Streaming Error', 'Connection lost. Please try again.');
          eventSource.close();
          eventSourceRef.current = null;
          setIsLoading(false);
        }
      });

    } catch (error) {
      console.error('Processing error:', error);
      setIsLoading(false);
      Alert.alert('Processing Error', `Failed to process audio: ${(error as any)?.message || (error as any)?.toString() || 'Unknown error occurred'}`);
    }
  };

  const formatLatency = (latency: number) => `${Math.round(latency)}ms`;
  const formatBackendLatency = (latency: number) => `${Math.round(latency * 1000)}ms`;

  const clearChat = () => {
    setTranscript('');
    setResponse('');
    setLatency(0);
    setLatencyStats(null);
    setRetrievedDocuments([]);
    setStreamingResponse('');
    setStreamingTranscript('');
    setStreamingStatus('');
    setStartTime(null);
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
              <Text style={styles.loadingText}>
                {streamingStatus || 'Processing your voice...'}
              </Text>
            </View>
          )}
        </View>

        {/* Chat Messages */}
        {(streamingTranscript || transcript || streamingResponse || response) && (
          <View style={styles.chatContainer}>
            {(streamingTranscript || transcript) && (
              <View style={styles.messageContainer}>
                <View style={styles.userMessage}>
                  <Ionicons name="person" size={20} color="#6366f1" style={styles.messageIcon} />
                  <Text style={styles.userMessageText}>{streamingTranscript || transcript}</Text>
                </View>
              </View>
            )}
            {(streamingResponse || response) && (
              <View style={styles.messageContainer}>
                <View style={styles.assistantMessage}>
                  <Ionicons name="chatbubble" size={20} color="#10b981" style={styles.messageIcon} />
                  <Text style={styles.assistantMessageText}>{streamingResponse || response}</Text>
                </View>
              </View>
            )}
            
            {/* Retrieved Documents */}
            {retrievedDocuments.length > 0 && (
              <View style={styles.retrievedDocsContainer}>
                <View style={styles.retrievedDocsHeader}>
                  <Ionicons name="document-text" size={16} color="#6b7280" />
                  <Text style={styles.retrievedDocsTitle}>Knowledge Sources</Text>
                </View>
                
                {retrievedDocuments.map((doc, index) => (
                  <View key={index} style={styles.docItem}>
                    <View style={styles.docInfo}>
                      <Text style={styles.docSource}>
                        {(doc.source || 'Unknown Source').replace('_chunk_', ' (Chunk ').replace(/(\d+)$/, '$1)')}
                        {doc.chunk_index && doc.total_chunks && (
                          <Text style={styles.chunkInfo}>
                            {' '}(Chunk {doc.chunk_index}/{doc.total_chunks})
                          </Text>
                        )}
                      </Text>
                      <Text style={styles.docOriginal}>
                        From: {(doc.original_file || 'Unknown File').replace('_chunk_', ' (Chunk ').replace(/(\d+)$/, '$1)')}
                      </Text>
                      {doc.token_count && (
                        <Text style={styles.tokenInfo}>
                          Tokens: {doc.token_count}
                        </Text>
                      )}
                    </View>
                    <View style={styles.docScore}>
                      <Text style={styles.scoreLabel}>Relevance</Text>
                      <Text style={styles.scoreValue}>
                        {(doc.similarity_score * 100).toFixed(1)}%
                      </Text>
                    </View>
                  </View>
                ))}
              </View>
            )}
            
            {/* Detailed Latency Statistics */}
            {latencyStats && (
              <View style={styles.latencyStatsContainer}>
                <View style={styles.latencyStatsHeader}>
                  <Ionicons name="analytics" size={16} color="#6b7280" />
                  <Text style={styles.latencyStatsTitle}>Performance Metrics</Text>
                </View>
                
                <View style={styles.latencyStatsGrid}>
                  <View style={styles.latencyStatItem}>
                    <Text style={styles.latencyStatLabel}>Audio Processing</Text>
                    <Text style={styles.latencyStatValue}>
                      {formatBackendLatency(latencyStats.audio_processing_latency)}
                    </Text>
                  </View>
                  
                  <View style={styles.latencyStatItem}>
                    <Text style={styles.latencyStatLabel}>Vector Search</Text>
                    <Text style={styles.latencyStatValue}>
                      {formatBackendLatency(latencyStats.retrieval_latency)}
                    </Text>
                  </View>
                  
                  <View style={styles.latencyStatItem}>
                    <Text style={styles.latencyStatLabel}>LLM Response</Text>
                    <Text style={styles.latencyStatValue}>
                      {formatBackendLatency(latencyStats.llm_latency)}
                    </Text>
                  </View>
                  
                  <View style={styles.latencyStatItem}>
                    <Text style={styles.latencyStatLabel}>Total Backend</Text>
                    <Text style={styles.latencyStatValue}>
                      {formatBackendLatency(latencyStats.total_latency)}
                    </Text>
                  </View>
                  
                  <View style={styles.latencyStatItem}>
                    <Text style={styles.latencyStatLabel}>Network + UI</Text>
                    <Text style={styles.latencyStatValue}>
                      {formatLatency(Math.max(0, latency - (latencyStats.total_latency * 1000)))}
                    </Text>
                  </View>
                  
                  <View style={styles.latencyStatItem}>
                    <Text style={styles.latencyStatLabel}>Total End-to-End</Text>
                    <Text style={styles.latencyStatValue}>
                      {formatLatency(latency)}
                    </Text>
                  </View>
                </View>
                
                <View style={styles.latencyBreakdown}>
                  <View style={styles.breakdownBar}>
                    <View 
                      style={[
                        styles.breakdownSegment, 
                        { 
                          backgroundColor: '#8b5cf6',
                          width: `${(latencyStats.audio_processing_latency / latencyStats.total_latency) * 100}%`
                        }
                      ]} 
                    />
                    <View 
                      style={[
                        styles.breakdownSegment, 
                        { 
                          backgroundColor: '#3b82f6',
                          width: `${(latencyStats.retrieval_latency / latencyStats.total_latency) * 100}%`
                        }
                      ]} 
                    />
                    <View 
                      style={[
                        styles.breakdownSegment, 
                        { 
                          backgroundColor: '#10b981',
                          width: `${(latencyStats.llm_latency / latencyStats.total_latency) * 100}%`
                        }
                      ]} 
                    />
                  </View>
                  <View style={styles.breakdownLegend}>
                    <View style={styles.legendItem}>
                      <View style={[styles.legendDot, { backgroundColor: '#8b5cf6' }]} />
                      <Text style={styles.legendText}>Audio</Text>
                    </View>
                    <View style={styles.legendItem}>
                      <View style={[styles.legendDot, { backgroundColor: '#3b82f6' }]} />
                      <Text style={styles.legendText}>Vector Search</Text>
                    </View>
                    <View style={styles.legendItem}>
                      <View style={[styles.legendDot, { backgroundColor: '#10b981' }]} />
                      <Text style={styles.legendText}>LLM Response</Text>
                    </View>
                  </View>
                </View>
              </View>
            )}
            
            {/* Fallback simple latency display */}
            {!latencyStats && latency > 0 && (
              <View style={styles.latencyContainer}>
                <Ionicons name="time" size={16} color="#6b7280" />
                <Text style={styles.latencyText}>Response time: {formatLatency(latency)}</Text>
              </View>
            )}
          </View>
        )}

        {/* Empty State */}
        {!streamingTranscript && !transcript && !streamingResponse && !response && !recording && !isLoading && (
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
          {(streamingTranscript || transcript || streamingResponse || response) && (
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
  latencyStatsContainer: {
    backgroundColor: '#ffffff',
    borderRadius: 12,
    padding: 16,
    marginTop: 16,
    borderWidth: 1,
    borderColor: '#e2e8f0',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.05,
    shadowRadius: 4,
    elevation: 2,
  },
  latencyStatsHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 12,
  },
  latencyStatsTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#1f2937',
    marginLeft: 6,
  },
  latencyStatsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
    marginBottom: 16,
  },
  latencyStatItem: {
    width: '48%',
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    backgroundColor: '#f8fafc',
    padding: 8,
    borderRadius: 8,
    marginBottom: 8,
  },
  latencyStatLabel: {
    color: '#6b7280',
    fontSize: 12,
    fontWeight: '500',
  },
  latencyStatValue: {
    color: '#1f2937',
    fontSize: 12,
    fontWeight: '600',
  },
  latencyBreakdown: {
    marginTop: 8,
  },
  breakdownBar: {
    flexDirection: 'row',
    height: 8,
    backgroundColor: '#f1f5f9',
    borderRadius: 4,
    overflow: 'hidden',
  },
  breakdownSegment: {
    height: '100%',
  },
  breakdownLegend: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    marginTop: 8,
  },
  legendItem: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  legendDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    marginRight: 6,
  },
  legendText: {
    color: '#6b7280',
    fontSize: 11,
    fontWeight: '500',
  },
  retrievedDocsContainer: {
    backgroundColor: '#ffffff',
    borderRadius: 12,
    padding: 16,
    marginTop: 16,
    borderWidth: 1,
    borderColor: '#e2e8f0',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.05,
    shadowRadius: 4,
    elevation: 2,
  },
  retrievedDocsHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 12,
  },
  retrievedDocsTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#1f2937',
    marginLeft: 6,
  },
  docItem: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    backgroundColor: '#f8fafc',
    padding: 12,
    borderRadius: 8,
    marginBottom: 8,
  },
  docInfo: {
    flex: 1,
    marginRight: 12,
  },
  docSource: {
    color: '#1f2937',
    fontSize: 14,
    fontWeight: '600',
    marginBottom: 2,
  },
  docOriginal: {
    color: '#6b7280',
    fontSize: 12,
  },
  docScore: {
    alignItems: 'flex-end',
  },
  scoreLabel: {
    color: '#6b7280',
    fontSize: 10,
    fontWeight: '500',
    marginBottom: 2,
  },
  scoreValue: {
    color: '#10b981',
    fontSize: 14,
    fontWeight: '600',
  },
  chunkInfo: {
    color: '#6b7280',
    fontSize: 12,
    fontWeight: '500',
  },
  tokenInfo: {
    color: '#6b7280',
    fontSize: 12,
    fontWeight: '500',
  },
});
