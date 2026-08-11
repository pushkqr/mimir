"""Warm worker — loads the docint pipeline once and processes PDFs on demand.

Protocol (stdin/stdout, one JSON object per line):
    Worker  →  parent:   {"status": "ready"}
    Parent  →  worker:   {"pdf_path": "/abs/path.pdf", "output_dir": "/abs/out/"}
    Worker  →  parent:   {"status": "ok", "en_txt": "/abs/out/name.pdf.en.txt"}
                      or {"status": "error", "message": "..."}

The worker stays alive between requests so the 200M IndicTrans2 model loaded by
doc_translator_hf is not re-loaded for every PDF.
"""

import json
import os
import shutil
import sys
from pathlib import Path


def _main():
    # word_recognizer.py is in the same working directory — importing it registers
    # the Vision factory so docint.load() can resolve the "word_recognizer" stage.
    import docint  # noqa
    import orgpedia  # noqa
    import word_recognizer  # noqa

    # Load the full 9-stage pipeline (CID → OCR → tables → paras → order_number
    # → translate → text_writer).  This is where the 200M model is loaded.
    viz = docint.load("src/writeTxt.yml")

    # Signal readiness
    print(json.dumps({"status": "ready"}), flush=True)

    input_dir = Path("input")
    output_dir = Path("output")
    input_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            pdf_path = req["pdf_path"]
            target_output_dir = req["output_dir"]

            pdf_name = os.path.basename(pdf_path)
            input_link = input_dir / pdf_name

            # Place the PDF where word_recognizer expects it: input/{name}
            if input_link.exists() or input_link.is_symlink():
                input_link.unlink()
            try:
                os.symlink(os.path.abspath(pdf_path), str(input_link))
            except OSError:
                # Symlink may fail on some systems; fall back to copy
                shutil.copy2(pdf_path, str(input_link))

            # Run the pipeline — stages 1-9, including translation + text_writer
            doc = viz(input_link)

            # doc.to_disk writes the .doc.json; text_writer already wrote .en.txt
            # and .mr.txt into the output_dir configured in writeTxt.yml ("output").
            doc_json_path = output_dir / (pdf_name + ".doc.json")
            doc.to_disk(doc_json_path)

            # Locate the .en.txt the text_writer stage produced
            en_txt_dst_candidates = [
                output_dir / (pdf_name + ".en.txt"),
                output_dir / (pdf_name.replace(".pdf", ".en.txt")),
                output_dir / (pdf_name.replace(".PDF", ".en.txt"))
            ]
            
            en_txt_src = next((c for c in en_txt_dst_candidates if c.exists() and c.stat().st_size > 0), None)

            if en_txt_src:
                os.makedirs(target_output_dir, exist_ok=True)
                en_txt_dst = os.path.join(target_output_dir, en_txt_src.name)
                shutil.copy2(str(en_txt_src), en_txt_dst)
                print(json.dumps({"status": "ok", "en_txt": en_txt_dst}), flush=True)
            else:
                out_files = os.listdir(output_dir) if output_dir.exists() else []
                print(json.dumps({
                    "status": "error",
                    "message": f"text_writer produced no .en.txt for {pdf_name}. output/ contains: {out_files}",
                }), flush=True)

            # Tidy up input symlink/copy (output files stay for debugging)
            if input_link.exists() or input_link.is_symlink():
                input_link.unlink()

        except Exception as exc:
            print(json.dumps({"status": "error", "message": str(exc)}), flush=True)


if __name__ == "__main__":
    _main()
