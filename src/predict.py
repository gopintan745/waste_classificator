import json
import torch
from PIL import Image
from src.transforms import get_val_transforms

class WasteClassifier:
    def __init__(self, model_path, class_index_path="class_index.json",
                 properties_path="src/waste_properties.json", device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.properties = json.load(open(properties_path))

        ckpt = torch.load(model_path, map_location=self.device, weights_only=False)
        self.classes = ckpt["classes"]
        self.model = self._build_model(ckpt)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        self.transform = get_val_transforms()

    def _build_model(self, ckpt):
        # Determine model architecture from checkpoint
        num_classes = len(self.classes)
        # Heuristic: look for "fc" in keys
        is_resnet = any("fc." in k for k in ckpt["model"].keys())
        if is_resnet:
            from src.models.transfer_model import build_transfer_model
            return build_transfer_model("resnet50", num_classes, pretrained=False)
        else:
            from src.models.custom_cnn import CustomWasteCNN
            return CustomWasteCNN(num_classes)

    @torch.no_grad()
    def predict(self, image_path, top_k=3, threshold=0.6):
        img = Image.open(image_path).convert("RGB")
        x = self.transform(img).unsqueeze(0).to(self.device)
        logits = self.model(x)
        probs = torch.softmax(logits, dim=1)[0]
        top_probs, top_idxs = probs.topk(top_k)

        result = {
            "top_k": [
                {"class": self.classes[i], "confidence": float(p)}
                for p, i in zip(top_probs, top_idxs)
            ]
        }

        # Use top-1 if confident, otherwise flag uncertainty
        top1_class = result["top_k"][0]["class"]
        top1_conf  = result["top_k"][0]["confidence"]

        if top1_conf < threshold:
            result["warning"] = f"Low confidence ({top1_conf:.1%}). The image may not be a recognized waste material or may contain multiple items."

        info = self.properties.get(top1_class, {})
        result["class_name"]   = top1_class
        result["recyclable"]   = info.get("recyclable", False)
        result["reusable"]     = info.get("reusable", False)
        result["safely_disposed"] = info.get("safely_disposed", False)
        result["hazardous"]    = info.get("hazardous", False)
        result["handling_steps"] = info.get("handling_steps", [])

        return result
