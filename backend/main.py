# main.py
from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pose_checker import analyze_pose
from elevenlabs.client import ElevenLabs
import uvicorn
import base64
import asyncio
import time

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Initialize ElevenLabs client
client = ElevenLabs(api_key="90c14bc3212b31e252833658f35cfb87b7076fa754f8a287e82b7757fec4de0e")

@app.get("/")
def root():
    return {"message": "Pose Detection API is running!"}

# ✅ FIXED: ElevenLabs text-to-speech endpoint with delay
@app.post("/speak")
async def speak(request: dict):
    """Generate speech using ElevenLabs"""
    text = request.get("text")
    delay = request.get("delay", 0)  # Get optional delay in seconds
    
    print(f"🔊 Speaking: {text}")

    if not text:
        raise HTTPException(status_code=400, detail="Text is required")
    
    try:
        # ✅ Add delay before generating speech
        if delay > 0:
            print(f"⏳ Waiting {delay} seconds before speaking...")
            await asyncio.sleep(delay)
        
        # ✅ Use turbo model (free tier compatible)
        audio = client.text_to_speech.convert(
            voice_id="21m00Tcm4TlvDq8ikWAM",  # Bella voice ID
            text=text,
            model_id="eleven_turbo_v2"
        )
        
        print("✅ Audio generated successfully")
        return StreamingResponse(audio, media_type="audio/mpeg")
    except Exception as e:
        print(f"❌ Speech error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Speech generation error: {str(e)}")

@app.websocket("/ws/stretch/{stretch_name}")
async def stretch_websocket(websocket: WebSocket, stretch_name: str):
    await websocket.accept()
    print(f"🟢 WebSocket connected for stretch: {stretch_name}")

    try:
        while True:
            data = await websocket.receive_text()
            
            if "," in data:
                data = data.split(",")[1] 
            data += "=" * (4 - len(data) % 4)
            
            try:
                image_bytes = base64.b64decode(data)
            except Exception as e:
                await websocket.send_json({"correct": False, "message": "Invalid image data"})
                continue

            pose_result = analyze_pose(image_bytes, stretch_name)

            if not pose_result["detected"]:
                await websocket.send_json({
                    "correct": False,
                    "status": "no_detection",
                    "message": "🔍 No person detected. Step into frame!"
                })
                continue
            
            await websocket.send_json(pose_result)

            if pose_result.get("correct") == True:
                print(f"✅ Stretch '{stretch_name}' completed!")
                break 

    except WebSocketDisconnect:
        print(f"🔴 Client disconnected from '{stretch_name}'")
    except Exception as e:
        print(f"⚠️ Error: {e}")
        await websocket.close()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
