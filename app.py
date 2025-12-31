"""
Raya & Mera - Enhanced Backend with Airport Knowledge
Abu Dhabi International Airport AI Assistants
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import json
from typing import Optional, List, Dict

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
    print(f"✅ Loaded airport knowledge: {sum(len(items) for items in AIRPORT_KNOWLEDGE.values())} items across {len(AIRPORT_KNOWLEDGE)} categories")
except Exception as e:
    print(f"⚠️ Warning: Could not load knowledge base: {e}")

# Character configurations
CHARACTERS = {
    'raya': {
        'name': 'Raya',
        'voice': 'en-GB-SoniaNeural',  # British English - Professional female
        'language': 'English',
        'emoji': '👩🇬🇧',
        'system_prompt': """You are Raya, a professional and articulate British AI assistant at Zayed International Airport (Abu Dhabi).

CORE IDENTITY:
- British English speaker with clear, professional manner
- Expert in airport services, facilities, and navigation
- Efficient, helpful, and articulate
- Knowledgeable about international travel procedures

AIRPORT KNOWLEDGE:
{knowledge_context}

RESPONSE STYLE:
- Clear, concise British English (2-4 sentences)
- Professional yet approachable
- Provide specific locations when available
- Use proper British spelling and phrasing
- Be direct and efficient

WHEN YOU DON'T KNOW:
- Be honest: "I'll need to verify that for you"
- Suggest asking airport information desk
- Provide general guidance based on standard practices

Remember: You're helping travelers navigate efficiently with British professionalism."""
    },
    'mera': {
        'name': 'Mera',
        'voice': 'ar-AE-FatimaNeural',  # Arabic (UAE) - Warm female
        'language': 'Arabic/English',
        'emoji': '👩🌟',
        'system_prompt': """You are Mera, a warm and welcoming Arabic AI assistant at Zayed International Airport (Abu Dhabi).

CORE IDENTITY:
- Warm Arabic hospitality and genuine care
- Expert in making travelers feel welcomed and comfortable
- Patient, understanding, and attentive
- Knowledgeable about Arabic culture and customs

AIRPORT KNOWLEDGE:
{knowledge_context}

RESPONSE STYLE:
- Warm, conversational tone (2-4 sentences)
- Show genuine care and hospitality
- Provide helpful, caring guidance
- Add warmth to every interaction
- Be patient and understanding

WHEN YOU DON'T KNOW:
- Be honest with warmth: "Let me find that out for you"
- Offer to help in other ways
- Provide caring guidance based on experience

Remember: You're welcoming travelers with genuine Arabic hospitality and warmth."""
    }
}

# Initialize Groq client
groq_client = None
try:
    from groq import Groq
    api_key = os.environ.get('GROQ_API_KEY')
    if api_key:
        groq_client = Groq(api_key=api_key)
        print("✅ Groq client initialized successfully")
    else:
        print("⚠️ No GROQ_API_KEY found")
except Exception as e:
    print(f"⚠️ Warning: Groq client initialization failed: {e}")

class ChatRequest(BaseModel):
    message: str
    character: str = 'raya'
    conversation_history: Optional[List[Dict]] = []

class ChatResponse(BaseModel):
    message: str
    character: str
    data_source: str
    has_airport_data: bool

def search_airport_knowledge(query: str):
    """
    Search airport knowledge base for relevant information
    Returns: (formatted_context, has_data)
    """
    if not AIRPORT_KNOWLEDGE:
        return "", False
    
    query_lower = query.lower()
    results = []
    
    # Search through all categories
    for category, items in AIRPORT_KNOWLEDGE.items():
        for item in items:
            # Check name
            if query_lower in item.get('name', '').lower():
                results.append(item)
                continue
            
            # Check keywords
            if any(keyword in query_lower for keyword in item.get('keywords', [])):
                results.append(item)
                continue
            
            # Check description
            if query_lower in item.get('description', '').lower():
                results.append(item)
    
    # Remove duplicates
    seen = set()
    unique_results = []
    for item in results:
        item_id = item.get('name', '') + item.get('location', '')
        if item_id not in seen:
            seen.add(item_id)
            unique_results.append(item)
    
    # Format results
    if unique_results:
        context = "📚 **Airport Database Information:**\n\n"
        for i, item in enumerate(unique_results[:3], 1):  # Top 3 results
            context += f"**{i}. {item['name']}**\n"
            if item.get('location'):
                context += f"📍 Location: {item['location']}\n"
            if item.get('description'):
                desc = item['description'][:250] + "..." if len(item['description']) > 250 else item['description']
                context += f"ℹ️ {desc}\n"
            if item.get('details'):
                context += f"🔍 {item['details']}\n"
            context += "\n"
        
        return context, True
    
    return "", False

async def get_llm_response(character: str, user_message: str, knowledge_context: str, has_data: bool) -> str:
    """Get response from Groq LLM"""
    if not groq_client:
        raise HTTPException(status_code=500, detail="LLM not available")
    
    # Get character config
    char_config = CHARACTERS.get(character, CHARACTERS['raya'])
    
    # Build system prompt with knowledge context
    system_prompt = char_config['system_prompt'].format(
        knowledge_context="Use the following airport database information to answer accurately:\n" + knowledge_context if has_data else "Use your general knowledge about airports."
    )
    
    # Build user message
    if has_data:
        full_message = f"{knowledge_context}\n\nUser Question: {user_message}\n\nPlease provide a natural, conversational response using the airport information above."
    else:
        full_message = user_message
    
    try:
        # Call Groq API
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": full_message}
            ],
            model="llama-3.3-70b-versatile",  # Use Groq's best model
            temperature=0.7,
            max_tokens=400,
            top_p=0.9,
        )
        
        return chat_completion.choices[0].message.content
        
    except Exception as e:
        print(f"❌ Groq API error: {e}")
        raise HTTPException(status_code=500, detail=f"LLM error: {str(e)}")

@app.get("/")
async def root():
    return {
        "status": "online",
        "service": "Raya & Mera Backend",
        "version": "3.0.0",
        "features": ["airport_knowledge", "llm_fallback", "dual_mode", "voice_ready"],
        "knowledge_loaded": len(AIRPORT_KNOWLEDGE) > 0,
        "knowledge_items": sum(len(items) for items in AIRPORT_KNOWLEDGE.values()) if AIRPORT_KNOWLEDGE else 0,
        "llm_available": groq_client is not None
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Handle chat requests with airport knowledge integration
    """
    try:
        # Step 1: Search airport knowledge
        knowledge_context, has_data = search_airport_knowledge(request.message)
        
        # Step 2: Get LLM response
        response_text = await get_llm_response(
            character=request.character,
            user_message=request.message,
            knowledge_context=knowledge_context,
            has_data=has_data
        )
        
        # Step 3: Return response
        return ChatResponse(
            message=response_text,
            character=request.character,
            data_source="database" if has_data else "llm",
            has_airport_data=has_data
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "airport_knowledge_loaded": len(AIRPORT_KNOWLEDGE) > 0,
        "knowledge_items": sum(len(items) for items in AIRPORT_KNOWLEDGE.values()) if AIRPORT_KNOWLEDGE else 0,
        "llm_available": groq_client is not None,
        "characters": list(CHARACTERS.keys())
    }

@app.get("/knowledge/search")
async def search_knowledge_endpoint(query: str):
    """Direct search endpoint for testing"""
    context, has_data = search_airport_knowledge(query)
    return {
        "query": query,
        "found": has_data,
        "context": context
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
