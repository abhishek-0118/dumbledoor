#!/usr/bin/env python3
"""
Setup script for the new authentication and chat features
"""

import os
import sys
import secrets
from pathlib import Path

def print_header(text):
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")

def print_step(step, text):
    print(f"\n{step}. {text}")

def check_environment():
    """Check and display required environment variables"""
    
    print_header("STARKFOUNDATION SETUP - AUTHENTICATION & CHAT FEATURES")
    
    print("\nThis script will help you set up the new features:")
    print("✅ Google OAuth authentication")
    print("✅ MongoDB chat history")
    print("✅ Conversational memory buffer")
    
    print_step(1, "Required Environment Variables")
    
    required_vars = {
        "GOOGLE_CLIENT_ID": "Google OAuth Client ID",
        "GOOGLE_CLIENT_SECRET": "Google OAuth Client Secret", 
        "JWT_SECRET_KEY": "JWT Secret Key for session management",
        "MONGODB_CONNECTION_STRING": "MongoDB connection string",
        "OPENAI_API_KEY": "OpenAI API key (existing)",
        "GITHUB_TOKEN": "GitHub token (existing)"
    }
    
    missing_vars = []
    
    for var, description in required_vars.items():
        value = os.getenv(var)
        if value:
            print(f"✅ {var}: {'*' * (len(value) - 4) + value[-4:] if len(value) > 4 else '***'}")
        else:
            print(f"❌ {var}: Missing - {description}")
            missing_vars.append(var)
    
    if missing_vars:
        print(f"\n⚠️  Missing {len(missing_vars)} environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        
        print_step(2, "How to set up missing variables")
        
        if "GOOGLE_CLIENT_ID" in missing_vars or "GOOGLE_CLIENT_SECRET" in missing_vars:
            print("\n🔐 Google OAuth Setup:")
            print("   1. Go to https://console.cloud.google.com/")
            print("   2. Create a new project or select existing one")
            print("   3. Enable Google+ API")
            print("   4. Go to 'Credentials' → 'Create Credentials' → 'OAuth client ID'")
            print("   5. Choose 'Web application'")
            print("   6. Add authorized redirect URI: http://localhost:8000/auth/google/callback")
            print("   7. Copy Client ID and Client Secret")
        
        if "JWT_SECRET_KEY" in missing_vars:
            print("\n🔑 JWT Secret Key:")
            jwt_secret = secrets.token_urlsafe(32)
            print(f"   Generated secret: {jwt_secret}")
            print(f"   Add to your environment: export JWT_SECRET_KEY='{jwt_secret}'")
        
        if "MONGODB_CONNECTION_STRING" in missing_vars:
            print("\n🍃 MongoDB Setup:")
            print("   Option 1 - Local MongoDB:")
            print("     - Install MongoDB: https://docs.mongodb.com/manual/installation/")
            print("     - Use: mongodb://localhost:27017")
            print("   Option 2 - MongoDB Atlas (Cloud):")
            print("     - Create account: https://www.mongodb.com/atlas")
            print("     - Get connection string from Atlas dashboard")
    
    else:
        print("\n✅ All environment variables are set!")
    
    print_step(3, "Install Dependencies")
    print("\nRun the following command to install new dependencies:")
    print("   pip install -r requirements.txt")
    
    print_step(4, "Start MongoDB (if using local)")
    print("\nIf using local MongoDB, start the service:")
    print("   # macOS with Homebrew:")
    print("   brew services start mongodb/brew/mongodb-community")
    print("   ")
    print("   # Linux (systemd):")
    print("   sudo systemctl start mongod")
    print("   ")
    print("   # Manual start:")
    print("   mongod --dbpath /path/to/data/directory")
    
    print_step(5, "Test the Setup")
    print("\nStart the server:")
    print("   python -m uvicorn app.api.server:app --reload --port 8000")
    print("\nTest endpoints:")
    print("   - Health check: http://localhost:8000/health")
    print("   - Google login: http://localhost:8000/auth/google/login")
    print("   - Create session: POST http://localhost:8000/chat/sessions")
    
    print_step(6, "API Usage Examples")
    
    print("\n🔐 Authentication Flow:")
    print("   1. GET /auth/google/login → Get authorization URL")
    print("   2. User visits URL and authorizes")
    print("   3. GET /auth/google/callback → Exchange code for tokens")
    print("   4. Use access_token in Authorization header: Bearer <token>")
    
    print("\n💬 Chat Session Flow:")
    print("   1. POST /chat/sessions → Create new session")
    print("   2. POST /chat/sessions/{id}/messages → Send messages")
    print("   3. GET /chat/sessions → List user's sessions")
    print("   4. GET /chat/sessions/{id}/messages → Get session history")
    
    print(f"\n{'='*60}")
    print("  Setup complete! Check the documentation for more details.")
    print(f"{'='*60}\n")

def create_example_requests():
    """Create example request scripts"""
    
    examples_dir = Path("examples")
    examples_dir.mkdir(exist_ok=True)
    
    # Authentication example
    auth_example = '''#!/usr/bin/env python3
"""
Example: Google OAuth Authentication
"""

import requests
import json

BASE_URL = "http://localhost:8000"

def test_auth_flow():
    """Test the complete authentication flow"""
    
    # Step 1: Get Google authorization URL
    print("1. Getting Google authorization URL...")
    response = requests.get(f"{BASE_URL}/auth/google/login")
    if response.status_code == 200:
        auth_data = response.json()
        print(f"Visit this URL to authorize: {auth_data['authorization_url']}")
        print("After authorization, you'll be redirected with a code parameter")
        
        # Step 2: Exchange code for tokens (you'll need to get the code from the callback)
        code = input("Enter the authorization code from the callback: ")
        if code:
            print("2. Exchanging code for tokens...")
            callback_response = requests.get(f"{BASE_URL}/auth/google/callback", params={"code": code})
            if callback_response.status_code == 200:
                tokens = callback_response.json()
                print("Authentication successful!")
                print(f"Access token: {tokens['access_token'][:20]}...")
                return tokens['access_token']
            else:
                print(f"Callback failed: {callback_response.text}")
    else:
        print(f"Failed to get authorization URL: {response.text}")
    
    return None

if __name__ == "__main__":
    access_token = test_auth_flow()
    if access_token:
        # Test authenticated endpoint
        headers = {"Authorization": f"Bearer {access_token}"}
        user_response = requests.get(f"{BASE_URL}/auth/me", headers=headers)
        if user_response.status_code == 200:
            user_info = user_response.json()
            print(f"Logged in as: {user_info['user']['name']} ({user_info['user']['email']})")
        else:
            print(f"Failed to get user info: {user_response.text}")
'''
    
    with open(examples_dir / "auth_example.py", "w") as f:
        f.write(auth_example)
    
    # Chat example
    chat_example = '''#!/usr/bin/env python3
"""
Example: Chat Session with Memory
"""

import requests
import json

BASE_URL = "http://localhost:8000"

def test_chat_session(access_token):
    """Test chat session with conversation memory"""
    
    headers = {"Authorization": f"Bearer {access_token}"}
    
    # Step 1: Create a new chat session
    print("1. Creating new chat session...")
    session_data = {
        "title": "Test Chat Session",
        "description": "Testing the new chat features",
        "repo_context": "citadel"
    }
    
    response = requests.post(f"{BASE_URL}/chat/sessions", headers=headers, json=session_data)
    if response.status_code == 200:
        session = response.json()["session"]
        session_id = session["id"]
        print(f"Created session: {session_id}")
        
        # Step 2: Send messages with conversation context
        messages = [
            "What is the purpose of this codebase?",
            "How does authentication work in this system?",
            "Can you explain the database models?",
            "What did we discuss about authentication earlier?"  # This should use conversation memory
        ]
        
        for i, message in enumerate(messages):
            print(f"\\n{i+2}. Sending message: {message}")
            
            message_data = {
                "message": message,
                "k": 10,
                "detailed_response": True
            }
            
            response = requests.post(
                f"{BASE_URL}/chat/sessions/{session_id}/messages", 
                headers=headers, 
                json=message_data
            )
            
            if response.status_code == 200:
                chat_response = response.json()
                print(f"Response: {chat_response['response'][:200]}...")
                print(f"Sources found: {chat_response['total_sources_found']}")
                print(f"Estimated cost: ${chat_response['estimated_cost']:.6f}")
            else:
                print(f"Message failed: {response.text}")
        
        # Step 3: Get conversation buffer stats
        print(f"\\n{len(messages)+2}. Getting buffer stats...")
        response = requests.get(f"{BASE_URL}/chat/sessions/{session_id}/buffer-stats", headers=headers)
        if response.status_code == 200:
            stats = response.json()["buffer_stats"]
            print(f"Buffer utilization: {stats['buffer_utilization']:.1%}")
            print(f"Total tokens: {stats['total_tokens']}")
            print(f"Key concepts: {', '.join(stats['key_concepts'])}")
        
        # Step 4: Get session history
        print(f"\\n{len(messages)+3}. Getting session history...")
        response = requests.get(f"{BASE_URL}/chat/sessions/{session_id}/messages", headers=headers)
        if response.status_code == 200:
            messages_history = response.json()["messages"]
            print(f"Total messages in session: {len(messages_history)}")
            for msg in messages_history[-2:]:  # Show last 2 messages
                print(f"  {msg['role']}: {msg['content'][:100]}...")
        
        return session_id
    else:
        print(f"Failed to create session: {response.text}")
        return None

if __name__ == "__main__":
    # You'll need an access token from the auth example
    access_token = input("Enter your access token: ")
    if access_token:
        test_chat_session(access_token)
'''
    
    with open(examples_dir / "chat_example.py", "w") as f:
        f.write(chat_example)
    
    print(f"Created example scripts in {examples_dir}/")

if __name__ == "__main__":
    check_environment()
    create_example_requests()
