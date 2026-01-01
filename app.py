"""
Raya & Mera - Complete Backend
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

app = FastAPI(title="Raya & Mera API", version="3.0.0")

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
    with open('airport_knowledge.json', 'r', encoding='utf-8') as f:
        AIRPORT_KNOWLEDGE = json.load(f)
    print(f"✅ Loaded airport knowledge: {sum(len(items) for items in AIRPORT_KNOWLEDGE.values())} items")
except Exception as e:
    print(f"⚠️ Warning: Could not load knowledge base: {e}")

CHARACTERS = {
    'raya': {
        'name': 'Raya',
        'voice': 'ar-AE-FatimaNeural',  # Arabic UAE voice
        'emoji': '🇦🇪',
        'system_prompt_arabic': """أنت رايا، مساعدة ذكية دافئة ومفيدة في مطار زايد الدولي (أبوظبي).

الهوية الأساسية:
- إماراتية، تتحدث العربية بطلاقة
- محترفة ودافئة ومرحبة
- خبيرة في خدمات المطار والمرافق والملاحة
- على دراية بأبوظبي وثقافة الإمارات

{knowledge_context}

أسلوب الرد:
- إجابات موجزة وقابلة للتنفيذ (2-4 جمل)
- قدمي مواقع محددة عند معرفتها
- كوني صبورة ومضيافة
- استخدمي اللغة العربية الطبيعية والمحادثة

عندما لا تعرفين:
- كوني صادقة: "دعيني أتحقق من ذلك"
- اقترحي السؤال عن موظفي المطار

تذكري: أنت تساعدين المسافرين بالضيافة الإماراتية الأصيلة.""",
        
        'system_prompt_english': """You are Raya, a warm and helpful AI assistant at Zayed International Airport (Abu Dhabi).

CORE IDENTITY:
- Emirati, warm and welcoming
- Expert in airport services, facilities, and navigation
- Knowledgeable about Abu Dhabi and UAE culture

{knowledge_context}

RESPONSE STYLE:
- Keep responses concise and actionable (2-4 sentences)
- Provide specific locations when known
- Be patient and hospitable
- Use natural, conversational English

WHEN YOU DON'T KNOW:
- Be honest: "Let me verify that"
- Suggest asking airport staff

Remember: You're helping travelers with genuine Emirati hospitality."""
    },
    'mera': {
        'name': 'Mera',
        'voice': 'en-GB-SoniaNeural',  # British English voice
        'emoji': '🌟',
        'system_prompt': """You are Mera, a professional and friendly AI assistant at Zayed International Airport (Abu Dhabi).

CORE IDENTITY:
- International perspective, fluent English speaker
- Professional, warm, and helpful
- Expert in airport services and international travel

{knowledge_context}

RESPONSE STYLE:
- Clear, concise English responses (2-4 sentences)
- Specific locations and practical directions
- Friendly and approachable tone
- ALWAYS respond in English, regardless of input language

WHEN YOU DON'T KNOW:
- Be honest: "Let me check that for you"
- Direct travelers to airport information desks

Remember: You ALWAYS respond in English only."""
    }
}

# Initialize Groq client
groq_client = None
try:
    api_key = os.environ.get('GROQ_API_KEY')
    if api_key:
        groq_client = Groq(api_key=api_key)
        print("✅ Groq client initialized")
    else:
        print("⚠️ No GROQ_API_KEY found")
except Exception as e:
    print(f"⚠️ Groq initialization failed: {e}")

class ChatRequest(BaseModel):
    message: str
    character: str = 'raya'
    conversation_history: Optional[List[Dict]] = []

class ChatResponse(BaseModel):
    message: str
    character: str
    data_source: str
    has_airport_data: bool
    audio_base64: Optional[str] = None

def detect_language(text: str) -> str:
    """Detect if text is Arabic or English"""
    arabic_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
    total_chars = len([c for c in text if c.isalpha()])
    
    if total_chars == 0:
        return 'english'
    
    return 'arabic' if arabic_chars / total_chars > 0.3 else 'english'

def search_airport_knowledge(query: str):
    """Search airport knowledge base"""
    if not AIRPORT_KNOWLEDGE:
        return "", False
    
    query_lower = query.lower()
    results = []
    
    for category, items in AIRPORT_KNOWLEDGE.items():
        for item in items:
            if query_lower in item.get('name', '').lower():
                results.append(item)
            elif any(keyword in query_lower for keyword in item.get('keywords', [])):
                results.append(item)
            elif query_lower in item.get('description', '').lower():
                results.append(item)
    
    # Remove duplicates
    seen = set()
    unique_results = []
    for item in results:
        item_id = item.get('name', '') + item.get('location', '')
        if item_id not in seen:
            seen.add(item_id)
            unique_results.append(item)
    
    if unique_results:
        context = "📚 **Airport Database Information:**\n\n"
        for i, item in enumerate(unique_results[:3], 1):
            context += f"**{i}. {item['name']}**\n"
            if item.get('location'):
                context += f"📍 Location: {item['location']}\n"
            if item.get('description'):
                desc = item['description'][:250] + "..." if len(item['description']) > 250 else item['description']
                context += f"ℹ️ {desc}\n"
            context += "\n"
        return context, True
    
    return "", False

async def generate_speech(text: str, voice: str) -> bytes:
    """Generate TTS audio using Edge TTS"""
    try:
        communicate = edge_tts.Communicate(text, voice)
        audio_data = b""
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data += chunk["data"]
        return audio_data
    except Exception as e:
        print(f"❌ TTS Error: {e}")
        return None

async def get_llm_response(character: str, user_message: str, knowledge_context: str, has_data: bool, input_language: str) -> str:
    """Get response from Groq LLM"""
    if not groq_client:
        raise HTTPException(status_code=500, detail="LLM not available")
    
    char_config = CHARACTERS.get(character)
    
    # For Raya: use language-specific prompt
    # For Mera: always use English prompt
    if character == 'raya':
        if input_language == 'arabic':
            system_prompt = char_config['system_prompt_arabic']
        else:
            system_prompt = char_config['system_prompt_english']
    else:  # mera
        system_prompt = char_config['system_prompt']
    
    # Add knowledge context
    system_prompt = system_prompt.format(
        knowledge_context="Use this airport database information:\n" + knowledge_context if has_data else ""
    )
    
    # Build user message
    if has_data:
        full_message = f"{knowledge_context}\n\nUser Question: {user_message}\n\nProvide a natural, conversational response."
    else:
        full_message = user_message
    
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": full_message}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.7,
            max_tokens=400,
        )
        
        return chat_completion.choices[0].message.content
        
    except Exception as e:
        print(f"❌ Groq error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {
        "status": "online",
        "service": "Raya & Mera Backend",
        "version": "3.0.0",
        "features": ["airport_knowledge", "llm_fallback", "tts_audio", "bilingual"],
        "knowledge_loaded": len(AIRPORT_KNOWLEDGE) > 0,
        "knowledge_items": sum(len(items) for items in AIRPORT_KNOWLEDGE.values()),
        "llm_available": groq_client is not None
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Handle chat with TTS audio generation"""
    try:
        # Detect input language (for Raya's response language)
        input_language = detect_language(request.message)
        print(f"📝 Input language detected: {input_language}")
        
        # Search airport knowledge
        knowledge_context, has_data = search_airport_knowledge(request.message)
        print(f"📚 Airport data found: {has_data}")
        
        # Get LLM response
        response_text = await get_llm_response(
            character=request.character,
            user_message=request.message,
            knowledge_context=knowledge_context,
            has_data=has_data,
            input_language=input_language
        )
        
        # Generate TTS audio - voices are CHARACTER-SPECIFIC, not language-specific
        if request.character == 'raya':
            # Raya: ALWAYS Arabic voice (even when responding in English)
            voice = 'ar-AE-FatimaNeural'
        else:
            # Mera: ALWAYS English voice
            voice = 'en-GB-SoniaNeural'
        
        print(f"🔊 Generating audio: {request.character} | {input_language} → {voice}")
        audio_bytes = await generate_speech(response_text, voice)
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8') if audio_bytes else None
        
        return ChatResponse(
            message=response_text,
            character=request.character,
            data_source="database" if has_data else "llm",
            has_airport_data=has_data,
            audio_base64=audio_base64
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "airport_knowledge_loaded": len(AIRPORT_KNOWLEDGE) > 0,
        "knowledge_items": sum(len(items) for items in AIRPORT_KNOWLEDGE.values()),
        "llm_available": groq_client is not None,
        "characters": ["raya", "mera"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
