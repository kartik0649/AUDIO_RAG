#!/usr/bin/env python3
"""
Test script for xAI SDK integration
"""

import os
from xai_sdk import Client
from xai_sdk.chat import user, system

def test_xai_integration():
    """Test xAI SDK integration"""
    
    # Check if API key is available
    api_key = os.getenv("GROK_API_KEY")
    if not api_key:
        print("❌ GROK_API_KEY environment variable not set")
        return False
    
    try:
        # Initialize xAI client
        print("🔧 Initializing xAI client...")
        client = Client(
            api_host="api.x.ai",
            api_key=api_key
        )
        print("✅ xAI client initialized successfully")
        
        # Test chat creation
        print("🔧 Creating chat session...")
        chat = client.chat.create(model="grok-3-mini-fast", temperature=0)
        print("✅ Chat session created successfully")
        
        # Test system message
        print("🔧 Adding system message...")
        chat.append(system("You are a helpful AI assistant."))
        print("✅ System message added successfully")
        
        # Test user message
        print("🔧 Adding user message...")
        chat.append(user("What is 2 + 2?"))
        print("✅ User message added successfully")
        
        # Test response
        print("🔧 Getting response...")
        response_text = str(chat)
        print(f"✅ Response received: {response_text}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during xAI integration test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 Testing xAI SDK Integration")
    print("=" * 40)
    
    success = test_xai_integration()
    
    if success:
        print("\n🎉 xAI integration test PASSED!")
    else:
        print("\n💥 xAI integration test FAILED!")
        print("\nTo fix this:")
        print("1. Install xAI SDK: pip install xai-sdk")
        print("2. Set GROK_API_KEY environment variable")
        print("3. Ensure you have access to xAI API") 