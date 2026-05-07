import json
import numpy as np
from ultralytics import YOLO

# Ayrı dosyadan tüm 103 sınıfı içe aktar
from class_mapping import FOODSEG103_CLASSES

class VisionAgent:
    def __init__(self, model_path="best.pt"):
        print(f"Görüntü Modeli ({model_path}) yükleniyor...")
        self.model = YOLO(model_path)
        
        # Artık o küçük manuel sözlük yerine doğrudan tam listeyi kullanıyoruz
        self.class_names = FOODSEG103_CLASSES

    def analyze_image(self, image_path):
        results = self.model.predict(image_path, conf=0.25, verbose=False)
        result = results[0]
        
        if result.masks is None:
            return json.dumps({
                "source": "vision_agent",
                "error": "No food detected", 
                "detected_items": []
            }, indent=4)
        
        masks = result.masks.data.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        
        total_food_pixels = 0
        item_pixel_counts = []
        
        for i, mask in enumerate(masks):
            pixel_count = np.sum(mask > 0)
            total_food_pixels += pixel_count
            
            class_id = int(classes[i])
            raw_class_name = result.names[class_id]
            
            # Sözlükte eşleşeni bul (örn: food_48 -> chicken duck)
            actual_name = self.class_names.get(raw_class_name, raw_class_name)
            
            item_pixel_counts.append({
                "name": actual_name,
                "pixel_count": int(pixel_count)
            })
            
        merged_pixels = {}
        for item in item_pixel_counts:
            name = item["name"]
            if name in merged_pixels:
                merged_pixels[name] += item["pixel_count"]
            else:
                merged_pixels[name] = item["pixel_count"]
                
        detected_items = []
        for name, p_count in merged_pixels.items():
            ratio = p_count / total_food_pixels if total_food_pixels > 0 else 0
            detected_items.append({
                "name": name,
                "pixel_ratio": round(ratio, 4)
            })
            
        final_output = {
            "source": "vision_agent",
            "detected_items": detected_items
        }
        
        return json.dumps(final_output, indent=4)

if __name__ == "__main__":
    TEST_IMAGE = "test_tabak.jpg" 
    
    agent = VisionAgent("best.pt")
    
    try:
        output = agent.analyze_image(TEST_IMAGE)
        print("\n--- VISION AGENT ÇIKTISI ---")
        print(output)
    except Exception as e:
        print(f"Bir hata oluştu: {e}")