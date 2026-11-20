# SafeGate AI: Smart Gate Access Control System
## Project Architecture & Implementation Roadmap

---

## 1. Project Overview & System Scope
**SafeGate AI** is an intelligent access control system designed to automate vehicle gate verification. It combines Real-Time Computer Vision (YOLOv8 object detection, EasyOCR Egyptian license plate recognition) with a structured Database Management Layer (`sqlite3`) and an Emergency Bypass Mechanism.

### Key Functional Requirements:
1. **Resident Identification:** Detect and recognize Egyptian license plates (Arabic letters and numerals) and verify access against resident records.
2. **Access Attempt Logging:** Record all entry attempts (Authorized, Visitor, Denied) with timestamps.
3. **Zero-Latency Emergency Vehicle Bypass:** Fast-track detection of emergency vehicles (Ambulance, Fire Truck, Police) using visual features prior to or independent of OCR processing.
4. **Administration & Manual Override:** Admin interface for onboarding residents, viewing logs, and manually triggering gate operations.

---

## 2. Design Patterns & Architectural Blueprint

To ensure scalability, testability, and clean separation of concerns, the following software design patterns are applied:

| Design Pattern | Application in SafeGate AI | Benefit |
| :--- | :--- | :--- |
| **Repository / Data Access Object (DAO)** | `DatabaseManager` class encapsulating SQL queries (`Residents`, `Access_Logs`). | Isolates SQL logic from Computer Vision and UI components. |
| **Pipeline Pattern** | Sequenced processing in `vision_system.py`: Frame Acquisition → Vehicle Detection → License Crop → OCR Extraction → Database Validation. | Simplifies debugging and allows swapping components (e.g., replacing EasyOCR with another OCR engine). |
| **Strategy Pattern** | Access Control Evaluation Strategies (e.g., Resident Plate Matching vs. Visual Emergency Vehicle Match). | Enables clean extension for future access rules (e.g., RFID tag integration or temporary visitor codes). |
| **Singleton / Resource Manager** | Database connection lifecycle and heavy ML model loaders (YOLOv8 & EasyOCR instances). | Prevents memory leaks and redundant model loading into GPU/RAM. |

---

## 3. Production Directory & File Structure

```text
Samar/
│
├── docs/                                 # Project documentation
│   ├── project_plan.md                  # System architecture, design patterns, and roadmap
│   └── setup_guide.md                   # Environment setup and dependency configuration
│
├── data/                                 # Persistent database & media assets
│   ├── safegate.db                      # SQLite database file
│   └── test_images/                     # Sample vehicle and license plate test images
│
├── models/                               # Machine Learning models & weights
│   ├── yolov8n.pt                       # Base YOLOv8 model weights
│   └── custom_emergency_yolo.pt         # Fine-tuned YOLO model for emergency vehicles
│
├── src/                                  # Core source code package
│   ├── __init__.py
│   │
│   ├── database/                        # Data Persistence Layer (DAO Pattern)
│   │   ├── __init__.py
│   │   └── db_manager.py                # DatabaseManager class & CRUD operations
│   │
│   ├── perception/                      # Computer Vision Layer (Pipeline Pattern)
│   │   ├── __init__.py
│   │   ├── detector.py                  # YOLOv8 vehicle & emergency detection module
│   │   ├── ocr_engine.py                # EasyOCR Egyptian Arabic plate parser
│   │   └── vision_pipeline.py           # Combined frame processor
│   │
│   ├── core/                            # Business Logic & Gate Controller (Strategy Pattern)
│   │   ├── __init__.py
│   │   ├── access_controller.py         # Decision logic (Resident vs. Emergency vs. Unknown)
│   │   └── logger.py                    # Application runtime logging
│   │
│   └── ui/                              # User Interface Layer (Admin & Dashboard)
│       ├── __init__.py
│       └── app.py                       # Streamlit / Gradio Dashboard for Admin
│
├── main.py                               # System entry point (runs camera & gate loop)
├── requirements.txt                      # Python dependencies list
└── README.md                             # Project overview and run instructions
```

---

## 4. Module Responsibilities & System Data Flow

```
   +-----------------------+
   |   Video / Camera Feed |
   +-----------+-----------+
               |
               v
   +-----------------------+
   |  YOLOv8 Detection     |  ---> Detects Emergency Vehicle? ---> [TRUE] ---> Instant Open Gate
   +-----------+-----------+
               | (Vehicle / Plate Cropped)
               v
   +-----------------------+
   |  EasyOCR Parser       |  ---> Extracts Arabic License Plate Text ("ا ب ج ١٢٣")
   +-----------+-----------+
               |
               v
   +-----------------------+
   |  Access Controller    |  ---> Queries DatabaseManager (`Residents`)
   +-----------+-----------+
               |
               +---> Allowed  --> Log attempt as 'Authorized Resident' --> Open Gate
               +---> Denied   --> Log attempt as 'Unknown/Visitor'     --> Keep Closed
```

---

## 5. Development Roadmap & Phases

### Phase 1: Database & Foundation (Completed)
- [x] Create `DatabaseManager` class using `sqlite3`.
- [x] Define `Residents` and `Access_Logs` schema.
- [x] Implement resident check and log insertion methods (`check_and_log_vehicle`).
- [x] Implement Egyptian dummy data seeding with Arabic plates (`seed_dummy_data`).

### Phase 2: Perception Layer Development (Current)
- [ ] Install PyTorch, Ultralytics YOLOv8, OpenCV, and EasyOCR.
- [ ] Create `perception/detector.py` for real-time bounding box detection.
- [ ] Implement Egyptian Arabic plate text normalization (mapping Eastern Arabic numerals `١٢٣` and standard Arabic letters).
- [ ] Build crop & preprocessing methods in OpenCV (contrast adjustment, sharpening).

### Phase 3: Core Orchestration & Strategy Logic
- [ ] Implement `access_controller.py` combining Perception output with Database validation.
- [ ] Integrate Zero-Latency Emergency Bypass (detecting ambulance/fire truck visual class).

### Phase 4: User Interface & Admin Dashboard
- [ ] Build a lightweight Admin GUI (`src/ui/app.py`) for:
  - Live gate camera feed view.
  - Adding/removing resident records.
  - Real-time access log history viewer.

---

## 6. Egyptian License Plate & Regional Considerations
1. **Character Set:** Egyptian plates feature 2 to 3 Arabic letters and 3 to 4 numbers.
2. **Numeral Formats:** Plates may present either Arabic-Indic numerals (`١٢٣٤`) or standard Latin numerals (`1234`). EasyOCR must handle both interchangeably.
3. **Emergency Vehicle Distinction:** Emergency vehicles (Ambulances, Police, Fire Services) carry distinct structural profiles and color schemes (e.g., grey/white/blue bands). Visual classification via YOLO avoids issues caused by non-standard or dirty plates during emergencies.