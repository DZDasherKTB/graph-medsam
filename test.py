from segment_anything import sam_model_registry

sam = sam_model_registry["vit_b"](checkpoint="weights/medsam_vit_b.pth")

print("MedSAM loaded successfully")
