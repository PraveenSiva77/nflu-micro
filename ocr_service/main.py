from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse
import shutil
import os
import uuid
import json
from typing import Optional
from ocr_validator import validate_document, TEMP_DIR

app = FastAPI()

# Ensure temp directory exists
os.makedirs(TEMP_DIR, exist_ok=True)

# Global Exception Handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    print(f"🔥 OCR Service Exception: {exc}")
    import traceback
    traceback.print_exc()
    return JSONResponse({
        "success": False,
        "error": "OCR Internal Server Error",
        "detail": str(exc)
    }, status_code=500)

@app.post("/validate")
async def validate_ocr(
    file: UploadFile = File(...),
    doc_type: str = Form("GENERIC"),
    expected_value: Optional[str] = Form(None)
):
    # Save uploaded file temporarily
    file_id = str(uuid.uuid4())
    file_ext = os.path.splitext(file.filename)[1]
    temp_file_path = os.path.join(TEMP_DIR, f"{file_id}{file_ext}")

    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Call the existing logic from ocr_validator.py
        # Since we modified validate_document to return a dictionary, we can use it directly
        output_dict = validate_document(temp_file_path, doc_type, expected_value)
        
        if not isinstance(output_dict, dict):
            raise HTTPException(status_code=500, detail="OCR logic failed to return a valid dictionary")
            
        return output_dict

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
