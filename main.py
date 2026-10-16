import uvicorn

if __name__ == "__main__":
    print("🌍 Starting Earth-Sim & Neural Brain Visualizer on http://localhost:8000 ...")
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
