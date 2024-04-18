from fastapi import FastAPI
from app.database import get_course

app = FastAPI()

@app.get("/courses/{course_id}")
async def read_course(course_id: int):
    course_data = get_course(course_id)
    if course_data:
        return course_data
    return {"error": "Course not found"}
