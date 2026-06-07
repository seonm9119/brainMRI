from fastapi import APIRouter, Query

from .volume_service import (
    get_modality_nifti_response,
    get_selected_cases_response
)


router = APIRouter(prefix="/api/brain-mri/segmentation", tags=["3D MRI Volume"])


@router.get("/cases")
def list_cases():
    return get_selected_cases_response()


@router.get("/cases/{case_id}/modality-nifti")
def get_modality_nifti(case_id: str, modality: str = Query("flair")):
    return get_modality_nifti_response(case_id, modality)
