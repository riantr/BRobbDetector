"""OCR sidecar using RapidOCR (ONNX-based PP-OCR, fast CPU inference).

Protocol:
  C# sends: image_path\n
  Python responds: {"text": "recognized text", "conf": 0.95}\n
  C# sends: QUIT\n to shut down
"""
import os, sys, json, contextlib

def main():
    devnull = open(os.devnull, 'w')
    with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
        from rapidocr_onnxruntime import RapidOCR
        ocr = RapidOCR()

    sys.stdout.write(json.dumps({"status": "ready"}) + "\n")
    sys.stdout.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        if line == "QUIT":
            break

        try:
            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                result, _ = ocr(line)
            texts = []
            confs = []
            if result:
                for item in result:
                    texts.append(item[1])
                    confs.append(float(item[2]))
            text = " ".join(texts)
            conf = sum(confs) / len(confs) if confs else 0.0
            sys.stdout.write(json.dumps({"text": text, "conf": round(conf, 3)}, ensure_ascii=False) + "\n")
        except Exception as e:
            sys.stdout.write(json.dumps({"text": "", "conf": 0.0, "error": str(e)}, ensure_ascii=False) + "\n")
        sys.stdout.flush()

    devnull.close()

if __name__ == "__main__":
    main()
