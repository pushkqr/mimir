import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import PlainTextResponse
from pathlib import Path

app = FastAPI(title="Orgpedia Transform Service")

# Load docint pipeline globally at startup so it stays warm
import docint
import orgpedia
import word_recognizer
viz = docint.load("src/writeTxt.yml")

input_dir = Path("input")
output_dir = Path("output")
input_dir.mkdir(exist_ok=True)
output_dir.mkdir(exist_ok=True)

@app.post("/transform")
async def transform_pdf(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
        
    pdf_name = file.filename
    input_path = input_dir / pdf_name
    
    with open(input_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
        
    try:
        # Run the pipeline (stages 1-9 including translation + text_writer)
        doc = viz(input_path)
        
        doc_json_path = output_dir / (pdf_name + ".doc.json")
        doc.to_disk(doc_json_path)
        
        # Locate the .en.txt the text_writer stage produced
        en_txt_dst_candidates = [
            output_dir / (pdf_name + ".en.txt"),
            output_dir / (pdf_name.replace(".pdf", ".en.txt")),
            output_dir / (pdf_name.replace(".PDF", ".en.txt"))
        ]
        
        en_txt_src = next((c for c in en_txt_dst_candidates if c.exists() and c.stat().st_size > 0), None)
        
        if not en_txt_src:
            out_files = os.listdir(output_dir) if output_dir.exists() else []
            raise HTTPException(
                status_code=500, 
                detail=f"text_writer produced no .en.txt for {pdf_name}. output/ contains: {out_files}"
            )
            
        with open(en_txt_src, "r", encoding="utf-8") as f:
            content = f.read()
            
        return PlainTextResponse(content)
        
    finally:
        if input_path.exists():
            input_path.unlink()
