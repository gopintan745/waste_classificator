"""
Waste material classifier with confidence-aware inference.
Designed around the per-class confusion insights from your results.
"""
import json
import torch
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.transforms import test_transforms
from src.models.custom_cnn import WasteClassifierCNN
from src.models.transfer_model import build_transfer_model


# ============================================
# CLASS GROUPING — Based on your confusion matrix analysis
# ============================================
# These three classes are visually similar and frequently confused:
# (Custom CNN had 13-18% confusion in this group)
SIMILAR_MATERIALS = {
    "glass": ["metal", "plastic"],
    "metal": ["glass", "plastic"],
    "plastic": ["glass", "metal"],
}

# Load full properties once at module load
PROPERTIES_PATH = Path(__file__).parent / "waste_properties.json"
PROPERTIES = json.load(open(PROPERTIES_PATH))


class WasteClassifier:
    """Wrapper around trained models with confidence-aware inference."""

    def __init__(
        self,
        model_path: str,
        model_arch: str = "transfer",   # "transfer" or "custom"
        device: str = "cuda",
        img_size: int = 256,
        confidence_threshold: float = 0.60,
        ambiguity_threshold: float = 0.20,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.confidence_threshold = confidence_threshold
        self.ambiguity_threshold = ambiguity_threshold
        self.img_size = img_size

        # Load model
        self.model, self.class_names = self._load_model(
            model_path, model_arch, device=self.device
        )
        self.model.eval()

        # Pre-built transform
        self.transform = test_transforms(img_size)

    def _load_model(self, model_path: str, model_arch: str, device: torch.device):
        """Load the trained model with the appropriate architecture."""
        ckpt = torch.load(model_path, map_location=device, weights_only=False)

        # Get class names from checkpoint
        class_names = ckpt.get("classes", 
                              ["cardboard", "glass", "metal", "paper", "plastic", "trash"])

        # Build the right architecture
        if model_arch == "transfer":
            # CRITICAL: must match the architecture used during training
            # Inspect checkpoint to determine which architecture
            state_dict = ckpt["model"]
            if any("features.0.0.weight" in k for k in state_dict.keys()):
                # EfficientNet
                model = build_transfer_model(
                    "efficientnet_b0",
                    num_classes=len(class_names),
                    pretrained=False,
                    dropout=0.34,
                    freeze_backbone=False,
                )
            elif any("layer1.0.conv1.weight" in k for k in state_dict.keys()):
                # ResNet
                model = build_transfer_model(
                    "resnet50",
                    num_classes=len(class_names),
                    pretrained=False,
                    dropout=0.45,
                    freeze_backbone=False,
                )
            else:
                # Default to EfficientNet
                model = build_transfer_model(
                    "efficientnet_b0",
                    num_classes=len(class_names),
                    pretrained=False,
                    dropout=0.34,
                    freeze_backbone=False,
                )
        else:
            # Custom CNN
            model = WasteClassifierCNN(
                num_classes=len(class_names),
                base_filters=16,
                dropout=0.12,
            )

        model.load_state_dict(ckpt["model"], strict=False)
        model.to(device)
        return model, class_names

    @torch.no_grad()
    def predict(self, image: Image.Image) -> Dict:
        """
        Predict the material class with confidence-aware logic.
        
        Returns a dict with:
          - top_class: str
          - confidence: float
          - top_k: list of (class, confidence) tuples
          - is_ambiguous: bool (top-1 close to top-2)
          - similar_materials: list of likely alternatives
          - properties: dict of material properties
          - warnings: list of warning strings
        """
        # Preprocess
        x = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)

        # Forward pass
        logits = self.model(x)
        probs = F.softmax(logits, dim=1)[0]
        probs_np = probs.cpu().numpy()

        # Get top-k
        top_k_idx = probs_np.argsort()[::-1][:3]
        top_k = [
            {"class": self.class_names[i], "confidence": float(probs_np[i])}
            for i in top_k_idx
        ]

        top1_class = top_k[0]["class"]
        top1_conf = top_k[0]["confidence"]
        top2_conf = top_k[1]["confidence"]

        # ============================================
        # UX-related logic
        # ============================================
        is_ambiguous = (top1_conf - top2_conf) < self.ambiguity_threshold

        # If the top-1 is below threshold, flag for review
        is_low_confidence = top1_conf < self.confidence_threshold

        # If the model is confused among similar materials, flag this
        similar_materials = []
        if top1_class in SIMILAR_MATERIALS:
            for i in range(1, len(top_k)):
                if top_k[i]["class"] in SIMILAR_MATERIALS[top1_class]:
                    similar_materials.append(top_k[i])

        # Properties for the top-1 class
        info = PROPERTIES.get(top1_class, {})

        # Generate warnings
        warnings = []
        if is_low_confidence:
            warnings.append(
                f"Low confidence ({top1_conf:.1%}). "
                f"The image may not be a recognized waste material."
            )
        if is_ambiguous and not is_low_confidence:
            warnings.append(
                f"Model is uncertain between {top1_class} ({top1_conf:.1%}) "
                f"and {top_k[1]['class']} ({top2_conf:.1%})."
            )
        if similar_materials:
            similar_str = ", ".join(
                f"{m['class']} ({m['confidence']:.1%})"
                for m in similar_materials
            )
            warnings.append(
                f"Note: {top1_class} can be visually similar to {similar_str}. "
                f"Verify the material if uncertain."
            )

        # ============================================
        # Track material class for analytics
        # ============================================
        if top1_class in ["glass", "metal", "plastic"]:
            warnings.append(
                "🔍 For glass, metal, or plastic items, check material "
                "markings (recycling code) for accurate classification."
            )

        return {
            "top_class": top1_class,
            "confidence": top1_conf,
            "top_k": top_k,
            "is_ambiguous": is_ambiguous,
            "is_low_confidence": is_low_confidence,
            "similar_materials": similar_materials,
            "properties": {
                "recyclable": info.get("recyclable", False),
                "reusable": info.get("reusable", False),
                "safely_disposed": info.get("safely_disposed", False),
                "hazardous": info.get("hazardous", False),
                "handling_steps": info.get("handling_steps", []),
            },
            "warnings": warnings,
        }


# ============================================
# USAGE EXAMPLE
# ============================================
if __name__ == "__main__":
    # Test the classifier
    import sys
    model_path = sys.argv[1] if len(sys.argv) > 1 else None
    if model_path is None:
        print("Usage: python classifier.py <model_path>")
        sys.exit(1)

    classifier = WasteClassifier(model_path, model_arch="transfer")
    print(f"Loaded model. Classes: {classifier.class_names}")
    print(f"Device: {classifier.device}")
    print(f"Confidence threshold: {classifier.confidence_threshold}")
    print(f"Ambiguity threshold: {classifier.ambiguity_threshold}")
