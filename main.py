import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from brain_mri_2d_slice_reconstruction.reconstruction.inference import (
    RECONSTRUCTION_CACHE_DIR,
    router as reconstruction_router
)
from brain_mri_3d_tumor_segmentation.segmentation.inference import (
    SEGMENTATION_CACHE_DIR,
    router as inference_router
)
from brain_mri_3d_tumor_segmentation.volume_3d.volume import (
    NIFTI_CACHE_DIR,
    router as volume_router
)


NIFTI_CACHE_DIR.mkdir(parents=True, exist_ok=True)
SEGMENTATION_CACHE_DIR.mkdir(parents=True, exist_ok=True)
RECONSTRUCTION_CACHE_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="brainMRI API")


def get_cors_origins():
    configured_origins = os.getenv("BRAINMRI_CORS_ORIGINS", "")
    if configured_origins.strip():
        return [origin.strip() for origin in configured_origins.split(",") if origin.strip()]

    return [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://192.168.0.11:3000",
        "http://3.226.20.82:30080"
    ]


app.add_middleware(
    CORSMiddleware,
    allow_origins=get_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

app.mount("/static/nifti-cache", StaticFiles(directory=NIFTI_CACHE_DIR), name="nifti-cache")
app.mount("/static/segmentation-cache", StaticFiles(directory=SEGMENTATION_CACHE_DIR), name="segmentation-cache")
app.mount("/static/reconstruction-cache", StaticFiles(directory=RECONSTRUCTION_CACHE_DIR), name="reconstruction-cache")
app.include_router(volume_router)
app.include_router(inference_router)
app.include_router(reconstruction_router)


@app.get("/api/health")
def health_check():
    return {
        "status": "ok",
        "service": "brainMRI"
    }
