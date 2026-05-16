# JARVIS AI Backend — Phase 1

> *"Just A Rather Very Intelligent System"*  
> Personal AI assistant backend for Mr Aryan — powered by Groq, FastAPI, and DuckDuckGo.

---

## Features

| Feature | Status |
|---|---|
| Hindi + English + Hinglish conversation | ✅ |
| Groq LLM integration (llama3-70b) | ✅ |
| Conversation memory (SQLite) | ✅ |
| Real-time web search (DuckDuckGo) | ✅ |
| JARVIS personality system | ✅ |
| REST API (FastAPI) | ✅ |
| WebSocket streaming architecture | ✅ Phase 2 ready |
| Proactive suggestions | ✅ (via system prompt) |
| Auto interactive API docs | ✅ `/docs` |

---

## Project Structure

```
jarvis-backend/
├── main.py                  # FastAPI app + lifespan + routers
├── config.py                # All settings via environment variables
├── requirements.txt         # Python dependencies
├── .env.example             # Template for your .env file
│
├── core/
│   ├── personality.py       # JARVIS system prompt, language detection, search trigger
│   └── context.py           # Context builder (history + search results → Groq messages)
│
├── database/
│   ├── db.py                # Async SQLAlchemy engine + session factory
│   └── models.py            # Conversation + Message ORM models
│
├── services/
│   ├── groq_service.py      # Groq API (standard + streaming completions)
│   ├── search_service.py    # DuckDuckGo web + news search
│   └── memory_service.py    # CRUD operations for conversations and messages
│
└── routers/
    ├── chat.py              # POST /chat  |  WS /chat/stream/{id}
    ├── memory.py            # GET/DELETE /memory/conversations/...
    ├── search.py            # POST /search/web  |  POST /search/news
    └── health.py            # GET /health  |  GET /health/status
```

---

## Quick Start

### 1. Clone / enter the project directory

```bash
cd jarvis-backend
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
```

Edit `.env` and add your **Groq API key** (free at https://console.groq.com):

```env
GROQ_API_KEY=gsk_your_key_here
GROQ_MODEL=llama3-70b-8192
```

### 4. Start the server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The server starts at **http://localhost:8000**  
Interactive API docs: **http://localhost:8000/docs**

---

## API Endpoints

### Chat

#### `POST /chat/`
Send a message to JARVIS and get a response.

```json
{
  "message": "Kya hai aaj ka weather Delhi mein?",
  "conversation_id": null,
  "force_search": false
}
```

**Response:**
```json
{
  "reply": "Mr Aryan, Delhi mein aaj mausam...",
  "conversation_id": "uuid-here",
  "language_detected": "hinglish",
  "used_web_search": true,
  "search_query": "Delhi weather today",
  "message_count": 2
}
```

Pass the returned `conversation_id` in subsequent requests to maintain context.

#### `WS /chat/stream/{conversation_id}`
WebSocket endpoint for streaming responses (Phase 2).

```json
// Send:
{"message": "Tell me about quantum computing"}

// Receive: text chunks streamed in real-time, then:
{"done": true, "conversation_id": "..."}
```

---

### Memory

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/memory/conversations` | List all conversations |
| `GET` | `/memory/conversations/{id}` | Get conversation + all messages |
| `GET` | `/memory/conversations/{id}/history` | Get context-ready history |
| `DELETE` | `/memory/conversations/{id}` | Delete a conversation |
| `DELETE` | `/memory/conversations/{id}/clear` | Clear messages only |

---

### Search

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/search/web` | DuckDuckGo web search |
| `POST` | `/search/news` | DuckDuckGo news search |

```json
{
  "query": "India cricket team 2025",
  "max_results": 5
}
```

---

### Health

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Liveness check |
| `GET` | `/health/status` | DB + Groq connectivity check |

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | **required** | Your Groq API key |
| `GROQ_MODEL` | `llama3-70b-8192` | Model to use |
| `DATABASE_URL` | `sqlite+aiosqlite:///./jarvis_memory.db` | SQLite path |
| `MAX_HISTORY_TURNS` | `20` | Conversation turns to keep in context |
| `MAX_SEARCH_RESULTS` | `5` | DuckDuckGo results per query |
| `APP_HOST` | `0.0.0.0` | Bind host |
| `APP_PORT` | `8000` | Bind port |
| `DEBUG` | `false` | Enable SQL logs + uvicorn reload |

---

## How It Works

1. **Message received** → language detected (Hindi / English / Hinglish)
2. **Search heuristic** → checks if web search is needed (keywords like "aaj", "latest", "price", etc.)
3. **DuckDuckGo search** → runs if triggered; results formatted into context
4. **History loaded** → last 20 conversation turns fetched from SQLite
5. **Context built** → system prompt + search results + history + current message
6. **Groq API called** → llama3-70b generates JARVIS's response
7. **Messages saved** → both turns persisted to SQLite for future context
8. **Response returned** → with conversation ID, language, and search metadata

---

## Phase 2 Roadmap

- [ ] WebSocket streaming UI (React / Next.js frontend)
- [ ] Voice input/output (Whisper STT + TTS)
- [ ] Task scheduling and reminders
- [ ] File upload and analysis
- [ ] Home automation integrations
- [ ] Multi-user support with authentication
