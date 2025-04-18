from fastapi import FastAPI

app = FastAPI(
    title="LSTM Server"
)

@app.get("/")
async def root():
    return {"message": "Hello World!"}

