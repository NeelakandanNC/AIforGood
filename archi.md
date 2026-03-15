```mermaid
graph TD
    %% Define Styles
    classDef frontend fill:#61dafb,stroke:#333,stroke-width:2px,color:#000
    classDef backend fill:#4caf50,stroke:#333,stroke-width:2px,color:#fff
    classDef agent fill:#ff9800,stroke:#333,stroke-width:2px,color:#fff
    classDef model fill:#9c27b0,stroke:#333,stroke-width:2px,color:#fff
    classDef llm fill:#0D47A1,stroke:#333,stroke-width:2px,color:#fff

    %% Frontend Components
    subgraph Client [React Frontend UI]
        Intake[Patient Intake Form]
        Dashboard[Real-time Dashboard]
        Queue[Patient Queue]
        Analytics[Analytics view]
    end

    %% Backend Services
    subgraph Server [FastAPI Backend]
        API[REST API /api/triage]
        SSE[SSE Stream /api/triage/stream]
        Store[(In-Memory Patient Store)]
    end

    %% ADK Pipeline
    subgraph Pipeline [Google ADK Multi-Agent Pipeline]
        Root[Root Sequential Agent]
        
        ClassAgent[Classification Agent]
        
        subgraph Council [Specialist Council - Parallel]
            Cardio[Cardiology Agent]
            Neuro[Neurology Agent]
            Pulmo[Pulmonology Agent]
            ER[Emergency Med Agent]
            GenMed[General Med Agent]
            Other[Other Specialty Agent]
        end
        
        CMO[CMO Agent]
    end

    %% External Dependencies
    XGBoost[(XGBoost ML Model)]
    Gemini[(Google Gemini LLM)]

    %% Connections
    Intake --> |POST Patient Data| API
    API --> Root
    
    Root --> ClassAgent
    ClassAgent <--> |Predict Risk| XGBoost
    
    ClassAgent --> Council
    Council <--> |Clinical Reasoning| Gemini
    Cardio & Neuro & Pulmo & ER & GenMed & Other --> CMO
    
    CMO <--> |Synthesize Verdict| Gemini
    CMO --> |CMO Verdict| SSE
    Council --> |Specialist Opinions| SSE
    ClassAgent --> |Risk Classification| SSE
    
    SSE --> |Live Updates| Dashboard
    API --> |Store Result| Store
    Store --> |Fetch Patients| Queue
    Store --> |Fetch Stats| Analytics

    %% Apply Styles
    class Intake,Dashboard,Queue,Analytics frontend
    class API,SSE,Store backend
    class Root,ClassAgent,Council,Cardio,Neuro,Pulmo,ER,GenMed,Other,CMO agent
    class XGBoost model
    class Gemini llm
