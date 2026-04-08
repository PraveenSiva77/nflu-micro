import uuid
import io
import os
from typing import Dict, Optional, List

from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import httpx
from motor.motor_asyncio import AsyncIOMotorClient
from fastapi.responses import JSONResponse

app = FastAPI(title="QR Scanning Microservice")

@app.get("/")
async def root():
    return {"message": "QR Service is running"}

class SessionCreate(BaseModel):
    user_id: str
    webhook_url: Optional[str] = None

# MongoDB Persistence Setup
DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL:
    print(f"🔌 Connecting to MongoDB for QR sessions...")
    db_client = AsyncIOMotorClient(DATABASE_URL)
    # Extract DB name from URL or use default
    db = db_client.get_default_database()
    sessions_col = db["qr_sessions"]
else:
    print("⚠️  DATABASE_URL not found, falling back to in-memory sessions (non-persistent)")
    sessions_col = None

# In-memory session store (Legacy/Fallback)
sessions: Dict[str, dict] = {}

# Resolve template path relative to this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")

if not os.path.exists(TEMPLATE_DIR):
    print(f"❌ WARNING: Template directory not found at {TEMPLATE_DIR}")

templates = Jinja2Templates(directory=TEMPLATE_DIR)

# Global Exception Handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    print(f"🔥 Global exception caught: {exc}")
    import traceback
    traceback.print_exc()

    # Determine if request is an API call or a Web Scanner load
    if request.url.path.startswith("/scan"):
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "session_id": "error",
                "status": "ERROR",
                "error_detail": str(exc)
            },
            status_code=500
        )
    
    # Return JSON for all other API endpoints (/sessions, /decode, etc.)
    return JSONResponse({
        "success": False,
        "error": "Internal Server Error",
        "detail": str(exc)
    }, status_code=500)

class ScanSubmit(BaseModel):
    scanned_data: str

async def get_session_data(session_id: str) -> Optional[dict]:
    """Helper to fetch session from MongoDB or Memory"""
    if sessions_col is not None:
        return await sessions_col.find_one({"session_id": session_id})
    return sessions.get(session_id)

async def save_session_data(session_id: str, data: dict):
    """Helper to save/update session in MongoDB or Memory"""
    if sessions_col is not None:
        await sessions_col.update_one(
            {"session_id": session_id},
            {"$set": data},
            upsert=True
        )
    else:
        sessions[session_id] = data


# ---------------------------------------------------------------------------
# Session endpoints (existing)
# ---------------------------------------------------------------------------

@app.post("/sessions")
async def create_session(session_data: SessionCreate):
    session_id = str(uuid.uuid4())
    print(f"🆕 Creating QR Session: {session_id} for user {session_data.user_id}")
    
    new_session = {
        "session_id": session_id,
        "user_id": session_data.user_id,
        "status": "PENDING",
        "scanned_data": None,
        "webhook_url": session_data.webhook_url
    }
    
    await save_session_data(session_id, new_session)
    return {"session_id": session_id, "scan_url": f"/scan/{session_id}"}


@app.get("/scan/{session_id}")
async def get_scanner(request: Request, session_id: str):
    print(f"🔍 Fetching QR Scanner for session: {session_id}")
    session = await get_session_data(session_id)
    
    if not session:
        print(f"❌ Session {session_id} not found")
        raise HTTPException(status_code=404, detail="Invalid Session")
        
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "session_id": session_id,
            "status": session["status"]
        }
    )


@app.post("/sessions/{session_id}/submit")
async def submit_scan(session_id: str, scan_data: ScanSubmit):
    session = await get_session_data(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Invalid Session")

    session["status"] = "COMPLETED"
    session["scanned_data"] = scan_data.scanned_data

    # Update state in persistent store
    await save_session_data(session_id, session)

    if session.get("webhook_url"):
        try:
            async with httpx.AsyncClient() as client:
                await client.post(session["webhook_url"], json={
                    "session_id": session_id,
                    "user_id": session["user_id"],
                    "scanned_data": scan_data.scanned_data
                })
        except Exception as e:
            print(f"Webhook notification failed: {e}")

    return {"message": "Scan successful!", "data": scan_data.scanned_data}


@app.get("/sessions/{session_id}/status")
async def get_status(session_id: str):
    session = await get_session_data(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Invalid Session")
    
    # Remove MongoDB internal ID for response
    if "_id" in session:
        del session["_id"]
    return session


# ---------------------------------------------------------------------------
# NEW: Decode QR from uploaded image
# ---------------------------------------------------------------------------

@app.post("/decode")
async def decode_qr_image(file: UploadFile = File(...)):
    """
    Accept an image file and decode any QR code found in it.
    Returns: { "success": bool, "data": str | null, "error": str | null }
    """
    content = await file.read()

    # Try pyzbar first (fastest, most reliable)
    try:
        from pyzbar.pyzbar import decode as pyzbar_decode
        from PIL import Image
        img = Image.open(io.BytesIO(content)).convert("RGB")
        barcodes = pyzbar_decode(img)
        if barcodes:
            data = barcodes[0].data.decode("utf-8")
            print(f"[QR Decode] pyzbar success: {data}")
            return {"success": True, "data": data, "error": None}
        else:
            print("[QR Decode] pyzbar: no barcodes found")
    except ImportError:
        print("[QR Decode] pyzbar library not installed")
    except Exception as e:
        print(f"[QR Decode] pyzbar error: {e}")

    # Fallback: OpenCV QRCodeDetector
    try:
        import cv2
        import numpy as np
        from PIL import Image
        img_pil = Image.open(io.BytesIO(content)).convert("RGB")
        img_np = np.array(img_pil)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        detector = cv2.QRCodeDetector()
        data, _, _ = detector.detectAndDecode(img_bgr)
        if data:
            print(f"[QR Decode] OpenCV success: {data}")
            return {"success": True, "data": data, "error": None}
        else:
            print("[QR Decode] OpenCV: no QR code found")
    except ImportError:
        print("[QR Decode] cv2 library not installed")
    except Exception as e:
        print(f"[QR Decode] OpenCV error: {e}")

    print("[QR Decode] Failed to find any QR codes in image")
    return {"success": False, "data": None, "error": "No QR code found in image"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
