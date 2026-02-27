# Ydhya

AI-powered triage assistant for India's sub-urban district hospitals. TriageAI uses a multi-agent pipeline to classify patient risk, gather specialist opinions, and deliver a consolidated verdict with workup plans — all in real-time via Server-Sent Events.

## Architecture

```
Patient Intake (React Frontend)
        │
        ▼
   POST /api/triage ──► FastAPI Backend
        │
        ▼
   SSE Stream ◄── Google ADK Multi-Agent Pipeline
                        │
                        ├── IngestAgent (parse & normalize input)
                        ├── ClassificationAgent (XGBoost ML risk prediction)
                        ├── SpecialistCouncil (parallel specialist evaluation)
                        │     ├── Cardiology Agent
                        │     ├── Neurology Agent
                        │     ├── Pulmonology Agent
                        │     ├── Emergency Medicine Agent
                        │     ├── General Medicine Agent
                        │     └── Other Specialty Agent
                        └── CMOAgent (final verdict & workup consolidation)
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React 19, Vite, Material UI, Recharts, Zustand |
| Backend | Python, FastAPI, Google ADK (Agent Development Kit) |
| ML Model | XGBoost classifier for risk prediction |
| LLM | Google Gemini (via ADK) |
| Streaming | Server-Sent Events (SSE) |

## Project Structure

```
pragyanxkanini/
├── backend/
│   ├── server.py              # Main FastAPI server (API + SSE)
│   ├── new_server.py          # WhatsApp integration server (Twilio)
│   ├── app/
│   │   ├── agent.py           # Root sequential agent
│   │   └── sub_agents/
│   │       ├── IngestAgent/        # Input parsing & normalization
│   │       ├── ClassificationAgent/ # XGBoost risk classification
│   │       ├── CMOAgent/           # Chief Medical Officer verdict
│   │       └── SpecialistCouncil/  # Parallel specialist evaluation
│   │           └── sub_agents/
│   │               ├── CardiologyAgent/
│   │               ├── NeurologyAgent/
│   │               ├── PulmonologyAgent/
│   │               ├── EmergencyMedicine/
│   │               ├── GeneralMedicine/
│   │               └── OtherSpecialityAgent/
│   ├── model/                 # Trained XGBoost model files
│   └── data/                  # Training/reference data
├── frontend/
│   ├── src/
│   │   ├── api/               # API client & SSE connection
│   │   ├── state/             # Zustand store
│   │   ├── layouts/           # Dashboard layout with sidebar
│   │   ├── pages/             # TriagePage, ResultPage, CouncilPage, QueuePage, AnalyticsPage
│   │   ├── components/
│   │   │   ├── common/        # RiskBadge, PriorityCircle, ActionChip, FlagChip
│   │   │   ├── intake/        # PatientForm, DocumentUpload
│   │   │   ├── stream/        # SSELogPanel (live pipeline stream)
│   │   │   ├── result/        # VerdictHeader, SafetyAlerts, WorkupTable
│   │   │   ├── council/       # CouncilRadar, SpecialistCard, ConsensusBar
│   │   │   ├── queue/         # Patient priority queue (DataGrid)
│   │   │   └── analytics/     # Stat cards, charts (Pie, Bar, Line)
│   │   ├── theme.js           # MUI theme configuration
│   │   └── utils/constants.js # Risk colors, symptom lists, specialist icons
│   └── vite.config.js         # Vite config with API proxy
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.11+
- Node.js 18+
- Google API Key (Gemini access)

### 1. Backend Setup

```bash
cd backend

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# Install dependencies
pip install fastapi uvicorn google-adk python-dotenv pydantic xgboost scikit-learn

# Set environment variable
echo "GOOGLE_API_KEY=your_key_here" > .env

And enable Vertex AI API from Google Cloud Console

# Start the server
python server.py
```

The backend runs on **http://localhost:8000**.

### 2. Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Start dev server
npm run dev
```

The frontend runs on **http://localhost:5173** with API requests proxied to the backend.

### 3. Open the App

Navigate to **http://localhost:5173** in your browser.

## Usage

### Triage Flow

1. **Patient Intake** — Fill in patient name, age, gender, vitals (BP, HR, Temperature in °F, SpO2), symptoms, and pre-existing conditions
2. **Live Stream** — Watch the SSE log panel as the pipeline processes through classification, specialist opinions, and CMO verdict in real-time
3. **View Results** — Navigate to the Result page for the full verdict with risk level, priority score, safety alerts, workup plan, and department routing
4. **Council View** — See the specialist council radar chart, individual specialist cards with relevance/urgency scores, flags, and differentials
5. **Patient Queue** — View all triaged patients sorted by priority score with filtering by risk level, department, and alerts
6. **Analytics** — Dashboard with stat cards, risk distribution pie chart, department load bar chart, trend lines, and alert frequency

### WhatsApp Integration

The backend also supports triage via WhatsApp (Twilio webhook). Start the WhatsApp server:

```bash
cd backend
python new_server.py
```

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/triage` | Start a triage session, returns `{ session_id, user_id }` |
| `GET` | `/api/triage/stream/{session_id}` | SSE stream of pipeline events |
| `GET` | `/api/dashboard/patients` | All triaged patient records |
| `GET` | `/api/dashboard/stats` | Aggregated dashboard statistics |
| `POST` | `/api/upload/document` | Upload a document for text extraction |

### SSE Event Types

| Event | Description |
|-------|-------------|
| `status` | Pipeline progress updates |
| `classification_result` | ML risk classification output |
| `specialist_opinion` | Individual specialist assessment (emitted per specialty) |
| `other_specialty_scores` | Relevance scores for other departments |
| `cmo_verdict` | Final enriched verdict with workup, alerts, and routing |
| `complete` | Pipeline finished |
| `error` | Error occurred |

## Features

- **Real-time SSE streaming** — Live pipeline updates with color-coded, timestamped log
- **Multi-agent AI pipeline** — Sequential + parallel agent architecture via Google ADK
- **ML risk classification** — XGBoost model for evidence-based risk prediction
- **Specialist council** — 5 specialist agents evaluate in parallel with radar visualization
- **Safety alerts** — Critical and warning flags surfaced from specialist assessments
- **Consolidated workup** — De-duplicated test recommendations grouped by priority (STAT/URGENT/ROUTINE)
- **Priority scoring** — 1-100 score combining urgency, relevance, and risk level
- **Department routing** — Primary/secondary department with referral recommendations
- **Patient queue** — Priority-sorted DataGrid with risk/department/alert filters
- **Analytics dashboard** — Stat cards, risk distribution, department load, trend charts
- **Responsive UI** — Sidebar collapses at 1024px, pages stack at 768px
- **WhatsApp integration** — Triage via WhatsApp messages and PDF uploads
