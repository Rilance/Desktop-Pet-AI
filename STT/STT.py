from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from faster_whisper import WhisperModel

app = FastAPI()

# Load the model
model = WhisperModel("large-v3", device="cpu", compute_type="int8")

@app.post("/transcribe/")
async def transcribe(file: UploadFile = File(...)):
    try:
        # Save the uploaded file temporarily
        with open(file.filename, "wb") as buffer:
            buffer.write(await file.read())
        
        # Transcribe the audio file
        segments, info = model.transcribe(file.filename)
        transcription_text = " ".join([segment.text for segment in segments])
        
        # Clean up the temporary file
        import os
        os.remove(file.filename)
        
        return JSONResponse(content={"text": transcription_text})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=9870)
