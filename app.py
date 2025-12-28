"""
Raya & Mira - Backend API (FIXED)
- Fixed Smart Mode voice routing bug
- Proper voice selection based on detected language
- Abu Dhabi Airport Assistant
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import asyncio
from typing import Optional, List, Dict
import edge_tts
from groq import Groq
import base64

app = FastAPI(title="Raya & Mira API", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CHARACTERS = {
    'raya': {
        'name': 'Raya',
        'voice': 'ar-AE-FatimaNeural',
        'language': 'Arabic',
        'emoji': '🇦🇪',
        'system_prompt': """You are Raya, a warm and welcoming Emirati airport assistant at Abu Dhabi International Airport. 
You specialize in helping Arabic-speaking travelers. You're knowledgeable about local culture, traditions, and the airport.
You're friendly, patient, and take pride in showing Emirati hospitality. Keep responses concise and helpful."""
    },
    'mira': {
        'name': 'Mira',
        'voice': 'en-GB-SoniaNeural',
        'language': 'English',
        'emoji': '🌍',
        'system_prompt': """You are Mira, an international airport assistant at Abu Dhabi International Airport.
You're warm, outspoken, and exceptionally helpful. You specialize in assisting English-speaking international travelers.
You're direct and clear in your communication, and assertive when needed. Keep responses concise and actionable."""
    }
}

groq_client = None
try:
    api_key = os.environ.get('GROQ_API_KEY')
    if api_key:
        groq_client = Groq(api_key=api_key)
except Exception as e:
    print(f"Warning: Groq client initialization failed: {e}")

class ChatRequest(BaseModel):
    message: str
    character: str = 'raya'
    mode: str = 'smart'
    history: Optional[List[Dict]] = []

class ChatResponse(BaseModel):
    character: str
    character_name: str
    emoji: str
    text_response: str
    audio_base64: Optional[str] = None
    voice: str

def detect_language(text: str) -> str:
    """Detect if text is primarily Arabic or English"""
    arabic_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
    total_chars = len([c for c in text if c.isalpha()])
    
    if total_chars == 0:
        return 'english'
    
    return 'arabic' if arabic_chars / total_chars > 0.3 else 'english'

def get_character_for_language(language: str) -> str:
    """Get appropriate character based on detected language"""
    return 'raya' if language == 'arabic' else 'mira'

async def generate_speech(text: str, voice: str) -> bytes:
    """Generate speech audio using Edge TTS"""
    try:
        communicate = edge_tts.Communicate(text, voice)
        audio_data = b""
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data += chunk["data"]
        return audio_data
    except Exception as e:
        print(f"TTS Error: {e}")
        return None

def get_llm_response(message: str, system_prompt: str, history: List[Dict]) -> str:
    """Get response from Groq LLM"""
    if not groq_client:
        return "Error: LLM service not configured. Please set GROQ_API_KEY."
    
    try:
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history)
        messages.append({"role": "user", "content": message})
        
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )
        
        return response.choices[0].message.content
    except Exception as e:
        print(f"LLM Error: {e}")
        return f"I'm having trouble processing that right now. Error: {str(e)}"

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "service": "Raya & Mira API",
        "version": "1.1.0",
        "characters": list(CHARACTERS.keys()),
        "fixes": ["Smart Mode voice routing", "Language detection improved"]
    }

@app.get("/characters")
async def get_characters():
    """Get available characters"""
    return {
        char_id: {
            'name': config['name'],
            'emoji': config['emoji'],
            'language': config['language'],
            'voice': config['voice']
        }
        for char_id, config in CHARACTERS.items()
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Process chat message and return response with audio - FIXED SMART MODE"""
    
    # Determine character based on mode
    if request.mode == 'smart':
        # Detect language from user message
        language = detect_language(request.message)
        character = get_character_for_language(language)
        print(f"Smart Mode: Detected {language} → Using {character}")
    else:
        character = request.character
    
    # Validate character
    if character not in CHARACTERS:
        raise HTTPException(status_code=400, detail="Invalid character")
    
    char_config = CHARACTERS[character]
    
    # Get LLM response
    text_response = get_llm_response(
        request.message,
        char_config['system_prompt'],
        request.history or []
    )
    
    # CRITICAL FIX: Use the character's correct voice for TTS
    # This was the bug - Smart Mode was using wrong voice!
    correct_voice = char_config['voice']
    
    print(f"Using voice: {correct_voice} for character: {character}")
    
    # Generate audio with CORRECT voice
    audio_bytes = await generate_speech(text_response, correct_voice)
    audio_base64 = base64.b64encode(audio_bytes).decode('utf-8') if audio_bytes else None
    
    return ChatResponse(
        character=character,
        character_name=char_config['name'],
        emoji=char_config['emoji'],
        text_response=text_response,
        audio_base64=audio_base64,
        voice=correct_voice  # Return which voice was actually used
    )

@app.get("/health")
async def health():
    """Detailed health check"""
    return {
        "status": "healthy",
        "groq_configured": groq_client is not None,
        "characters_available": len(CHARACTERS),
        "version": "1.1.0",
        "bug_fixes": ["Smart Mode voice routing fixed"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
