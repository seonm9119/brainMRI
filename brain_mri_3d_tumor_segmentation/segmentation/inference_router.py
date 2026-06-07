from fastapi import APIRouter, Query

from .inference_service import get_prediction_response


router = APIRouter(prefix="/api/brain-mri/segmentation", tags=["Segmentation Inference"])


@router.get("/cases/{case_id}/prediction")
def get_prediction(case_id: str, model: str = Query("assignment")):
    return get_prediction_response(case_id, model)
