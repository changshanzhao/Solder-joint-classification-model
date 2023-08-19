from ultralytics import YOLO

model = YOLO('best.pt')

success=model.export(format="onnx", simplify=True, opset=12)