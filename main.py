from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from brain_mri_3d_tumor_segmentation.segmentation.inference_router import router as inference_router
from brain_mri_3d_tumor_segmentation.segmentation.inference_service import SEGMENTATION_CACHE_DIR
from brain_mri_3d_tumor_segmentation.volume_3d.volume_router import router as volume_router
from brain_mri_3d_tumor_segmentation.volume_3d.volume_service import NIFTI_CACHE_DIR


NIFTI_CACHE_DIR.mkdir(parents=True, exist_ok=True)
SEGMENTATION_CACHE_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="brainMRI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://192.168.0.11:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

app.mount("/static/nifti-cache", StaticFiles(directory=NIFTI_CACHE_DIR), name="nifti-cache")
app.mount("/static/segmentation-cache", StaticFiles(directory=SEGMENTATION_CACHE_DIR), name="segmentation-cache")
app.include_router(volume_router)
app.include_router(inference_router)


@app.get("/api/health")
def health_check():
    return {
        "status": "ok",
        "service": "brainMRI"
    }
