"""
Raya & Mira - Enhanced Backend with Airport Knowledge
Abu Dhabi International Airport AI Assistants
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
import json

app = FastAPI(title="Raya & Mira API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load airport knowledge base
AIRPORT_KNOWLEDGE = {}
try:
    import os
    # Try multiple possible paths
    possible_paths = [
        'airport_knowledge_compact.json',
        './airport_knowledge_compact.json',
        '/opt/render/project/src/airport_knowledge_compact.json'
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"Found knowledge base at: {path}")
            with open(path, 'r', encoding='utf-8') as f:
                AIRPORT_KNOWLEDGE = json.load(f)
                total_items = sum(len(items) for items in AIRPORT_KNOWLEDGE.values())
                print(f"✅ Loaded {total_items} airport knowledge items")
                break
    
    if not AIRPORT_KNOWLEDGE:
        print("❌ ERROR: Could not find airport_knowledge_compact.json!")
        print(f"Current directory: {os.getcwd()}")
        print(f"Files available: {os.listdir('.')}")
        
except Exception as e:
    print(f"❌ ERROR loading knowledge base: {e}")

CHARACTERS = {
    'raya': {
        'name': 'Raya',
        'voice': 'ar-AE-FatimaNeural',
        'language': 'Arabic',
        'emoji': '🇦🇪',
        'system_prompt': """You are Raya, a warm and helpful AI assistant at Abu Dhabi International Airport (Zayed International Airport).

CORE IDENTITY:
- Emirati, fluent in Arabic
- Professional yet warm and welcoming
- Expert in airport services, facilities, and navigation
- Knowledgeable about Abu Dhabi and UAE culture

AIRPORT KNOWLEDGE AREAS:
- Terminals, gates, and navigation
- Restaurants, cafes, and dining options (46 locations)
- Shops and duty-free (64 locations)
- Transportation (parking, taxis, metro, shuttles)
- Services (lounges, prayer rooms, medical, charging stations)
- Special assistance and accessibility
- Travel procedures (check-in, security, immigration, customs)

RESPONSE STYLE:
- Keep responses concise and actionable (2-4 sentences)
- Provide specific locations when known
- Offer helpful suggestions proactively
- Use natural, conversational Arabic
- Be patient and hospitable

WHEN YOU DON'T KNOW:
- Be honest: "دعني أتحقق من ذلك" (Let me verify that)
- Suggest asking airport staff
- Provide general guidance based on airport best practices

Remember: You're helping travelers navigate the airport efficiently and comfortably."""
    },
    'mira': {
        'name': 'Mira',
        'voice': 'en-GB-SoniaNeural',
        'language': 'English',
        'emoji': '🌍',
        'system_prompt': """You are Mira, a friendly and efficient AI assistant at Abu Dhabi International Airport (Zayed International Airport).

CORE IDENTITY:
- International perspective, fluent English speaker
- Professional, warm, and outspoken when needed
- Expert in airport services and international travel
- Helpful and direct in communication

AIRPORT KNOWLEDGE AREAS:
- Terminals, gates, and wayfinding
- Restaurants, cafes, and dining (46 locations)
- Shopping and duty-free (64 stores)
- Ground transportation (parking, taxis, metro, rental cars)
- Airport services (lounges, medical, prayer rooms, facilities)
- Accessibility and special assistance
- Immigration, customs, and travel procedures

RESPONSE STYLE:
- Clear, concise responses (2-4 sentences)
- Specific locations and practical directions
- Proactive helpful suggestions
- Assertive when safety or procedures matter
- Friendly and approachable tone

WHEN YOU DON'T KNOW:
- Be honest: "Let me check that for you"
- Direct travelers to airport information desks
- Provide general guidance based on standard airport practices

Remember: You're helping international travelers have a smooth, stress-free experience."""
    }
}

groq_client = None
try:
    api_key = os.environ.get('GROQ_API_KEY')
    if api_key:
        groq_client = Groq(api_key=api_key)
        print("Groq client initialized successfully")
except Exception as e:
    print(f"Warning: Groq client initialization failed: {e}")

class ChatRequest(BaseModel):
    message: str
    character: str = 'raya'
    mode: str = 'dual'
    history: Optional[List[Dict]] = []

class ChatResponse(BaseModel):
    character: str
    character_name: str
    emoji: str
    text_response: str
    audio_base64: Optional[str] = None
    voice: str

def search_knowledge(query: str, category: Optional[str] = None) -> List[Dict]:
    """
    Search airport knowledge base for relevant information
    Returns list of matching items
    """
    results = []
    query_lower = query.lower()
    
    # Simple keyword matching (can be enhanced with better search later)
    search_categories = [category] if category else AIRPORT_KNOWLEDGE.keys()
    
    for cat in search_categories:
        if cat in AIRPORT_KNOWLEDGE:
            for item in AIRPORT_KNOWLEDGE[cat]:
                # Check if query matches name, description, or keywords
                if (query_lower in item.get('name', '').lower() or
                    query_lower in item.get('description', '').lower() or
                    any(query_lower in kw.lower() for kw in item.get('keywords', []))):
                    results.append({
                        'category': cat,
                        'name': item.get('name'),
                        'description': item.get('description', '')[:200],  # Truncate
                        'location': item.get('location', 'Check airport map'),
                        'details': item.get('details', '')
                    })
                    
                    if len(results) >= 5:  # Limit results
                        return results
    
    return results

def enhance_prompt_with_context(message: str, character_prompt: str) -> str:
    """
    Enhance system prompt with relevant airport knowledge based on user query
    """
    # Search for relevant knowledge
    knowledge_items = search_knowledge(message)
    
    if knowledge_items:
        context = "\n\nRELEVANT AIRPORT INFORMATION:\n"
        for item in knowledge_items[:3]:  # Top 3 results
            context += f"- {item['name']} ({item['category']}): {item['location']}\n"
            if item.get('details'):
                context += f"  Details: {item['details']}\n"
        
        return character_prompt + context
    
    return character_prompt

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
    """Get response from Groq LLM with airport knowledge context"""
    if not groq_client:
        return "Error: LLM service not configured. Please set GROQ_API_KEY."
    
    try:
        # Enhance prompt with relevant airport knowledge
        enhanced_prompt = enhance_prompt_with_context(message, system_prompt)
        
        messages = [{"role": "system", "content": enhanced_prompt}]
        messages.extend(history[-10:])  # Last 10 messages for context
        messages.append({"role": "user", "content": message})
        
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            temperature=0.7,
            max_tokens=300  # Keep responses concise
        )
        
        return response.choices[0].message.content
    except Exception as e:
        print(f"LLM Error: {e}")
        return f"I'm having trouble processing that right now. Please try rephrasing or ask airport staff for assistance."

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "service": "Raya & Mira API",
        "version": "2.0.0",
        "characters": list(CHARACTERS.keys()),
        "features": [
            "Airport knowledge base (256 items)",
            "Smart context-aware responses",
            "Bilingual support (Arabic/English)",
            "Text-to-speech audio"
        ]
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

@app.get("/knowledge/stats")
async def knowledge_stats():
    """Get airport knowledge base statistics"""
    stats = {}
    total = 0
    for category, items in AIRPORT_KNOWLEDGE.items():
        count = len(items)
        stats[category] = count
        total += count
    
    return {
        "total_items": total,
        "categories": stats,
        "status": "loaded" if AIRPORT_KNOWLEDGE else "empty"
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Process chat message with airport knowledge integration"""
    
    # Determine character
    if request.mode == 'smart':
        language = detect_language(request.message)
        character = get_character_for_language(language)
        print(f"Smart Mode: Detected {language} → Using {character}")
    else:
        character = request.character
    
    if character not in CHARACTERS:
        raise HTTPException(status_code=400, detail="Invalid character")
    
    char_config = CHARACTERS[character]
    
    # Get LLM response with airport knowledge
    text_response = get_llm_response(
        request.message,
        char_config['system_prompt'],
        request.history or []
    )
    
    # Generate audio
    correct_voice = char_config['voice']
    print(f"Generating audio with voice: {correct_voice}")
    
    audio_bytes = await generate_speech(text_response, correct_voice)
    audio_base64 = base64.b64encode(audio_bytes).decode('utf-8') if audio_bytes else None
    
    return ChatResponse(
        character=character,
        character_name=char_config['name'],
        emoji=char_config['emoji'],
        text_response=text_response,
        audio_base64=audio_base64,
        voice=correct_voice
    )

@app.get("/health")
async def health():
    """Detailed health check"""
    return {
        "status": "healthy",
        "groq_configured": groq_client is not None,
        "characters_available": len(CHARACTERS),
        "knowledge_items": sum(len(items) for items in AIRPORT_KNOWLEDGE.values()),
        "version": "2.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
