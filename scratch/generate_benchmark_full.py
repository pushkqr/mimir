"""
Document-level benchmark generator for the full corpus.
Generates 2-4 diverse questions per document using Gemini, fed the full doc text.

Usage:
    python scratch/generate_benchmark_full.py [--limit N] [--target N] [--out benchmark/benchmark_100.json]

Strategy:
  - Feed FULL document text (not chunks) to Gemini
  - Ask for officer-perspective questions with explicit diversity requirements
  - Questions must stand alone with zero reference to "this document" (officers
    querying a RAG assistant have not seen the source file, so self-referential
    phrasing makes a question unanswerable in isolation and breaks eval)
  - Incrementally saves so it can be resumed if interrupted
  - Deduplicates across documents
  - Backfills from remaining docs if skips/filtering leave the set short of target
"""
import os, sys, json, glob, time, re, random, argparse
sys.path.insert(0, os.getcwd())
sys.stdout.reconfigure(encoding="utf-8")
from dotenv import load_dotenv
load_dotenv()

import pymupdf4llm
from google import genai
from google.genai import types
from core.utils import get_genai_client

GENERATION_PROMPT = """You are helping build a benchmark test set for a Government Document RAG system used by Maharashtra government officers.

The officer asking these questions has NOT seen this document. They are typing a query into a search assistant cold, with no idea which file will answer it. This means every question must stand completely alone with zero reference to "this document," "the above," "the attached," "this circular," "this GR," "ha parishipatra," "ya nirnayat," or any phrase that assumes the reader already has the document open. Instead, name the specific subject, scheme, department, date, or GR number so the question makes sense in total isolation.

BAD: "What is the GR number of this circular?"
GOOD: "What's the GR number for the Kolhapur engineering college staffing order?"

BAD: "या दस्तऐवजात कोणत्या तारखेला परिपत्रक जारी झाले?"
GOOD: "कोल्हापूर अभियांत्रिकी महाविद्यालयाचं परिपत्रक कोणत्या तारखेला निघालं?"

BAD: "इस दस्तावेज़ में कौन सी योजना बताई गई है?"
GOOD: "कोल्हापूर इंजीनियरिंग कॉलेज की स्टाफिंग योजना में क्या-क्या शामिल है?"

Given the following government document text, generate exactly {n_questions} test questions. They must read like something a real officer typed into a search bar, conversational and practical, never academic.

Required mix:
1. One SIMPLE factual question in English, answerable from a single sentence.
2. One question in natural MARATHI (Devanagari), phrased how an officer actually talks, not a translation.
3. One question in natural HINDI (Devanagari), phrased how a Hindi-speaking officer would actually ask it, not a translation of the English or Marathi question.
4. One question that connects two facts stated in different parts of the document (not deep multi-hop reasoning, just two facts an officer would naturally ask together, e.g. "who approved X and when").
{extra_requirements}

RULES:
- Every question must name the specific subject/scheme/entity it's about. Never rely on document context to disambiguate.
- Answerable from this document's text only, don't invent facts.
- expected_answer must be written in the same language as the query.
- expected_terms must be exact substrings copied from the document text, not paraphrased, they are used for exact-match scoring.
- Keep phrasing colloquial, not legalese.
- Marathi and Hindi questions must be genuinely distinct in wording and phrasing, not the same question with words swapped between the two languages.

Return ONLY a valid JSON array with this structure (no markdown, no explanation):
[
  {{
    "query": "the question text",
    "expected_answer": "the ideal answer, same language as query",
    "expected_terms": ["verbatim", "terms", "from", "document"],
    "category": "simple_english|marathi_query|hindi_query|complex_english|gr_number_lookup"
  }}
]

DOCUMENT TEXT:
---
{doc_text}
---"""

NOT_FOUND_QUESTIONS = [
    {
        "query": "What is the pension amount for retired government professors in Maharashtra?",
        "expected_answer": "The documents available in the system do not contain information about pension amounts for retired government professors. Please check the Finance Department rules.",
        "expected_terms": ["not found", "not available"],
        "category": "not_found",
        "source_doc": None
    },
    {
        "query": "शासकीय महाविद्यालयातील विद्यार्थ्यांसाठी शिष्यवृत्तीची रक्कम किती आहे?",
        "expected_answer": "उपलब्ध दस्तऐवजांमध्ये शासकीय महाविद्यालयातील शिष्यवृत्तीबद्दल माहिती नाही.",
        "expected_terms": ["not found", "not available"],
        "category": "not_found",
        "source_doc": None
    },
    {
        "query": "Government engineering college mein admission ke liye minimum percentage kya chahiye?",
        "expected_answer": "The retrieved documents do not contain information about minimum percentage requirements for engineering college admissions.",
        "expected_terms": ["not found", "not available"],
        "category": "not_found",
        "source_doc": None
    },
    {
        "query": "What is the hostel fee for students at Government Engineering College Kolhapur?",
        "expected_answer": "The documents in the system do not contain information about hostel fees. The GR on the new Kolhapur Government Engineering College only covers course approval and staffing plans.",
        "expected_terms": ["not found", "not available"],
        "category": "not_found",
        "source_doc": None
    },
    {
        "query": "What are the medical leave rules for teaching staff in Maharashtra government colleges?",
        "expected_answer": "The retrieved documents do not contain information about medical leave rules for teaching staff.",
        "expected_terms": ["not found", "not available"],
        "category": "not_found",
        "source_doc": None
    },
]

SELF_REFERENCE_PATTERNS = [
    r"\bthis (document|circular|order|gr|notification|scheme|resolution)\b",
    r"\bthe above\b",
    r"\bthe attached\b",
    r"\bthe given (document|circular|order)\b",
    r"\babove[- ]mentioned\b",
    r"या (दस्तऐवजा|परिपत्रका|आदेशा|शासन निर्णया|अधिसूचने)त",
    r"ह्या (दस्तऐवजा|परिपत्रका|आदेशा|शासन निर्णया)त",
    r"वरील (दस्तऐवज|परिपत्रक|आदेश|शासन निर्णय)",
    r"इस (दस्तावेज़|दस्तावेज|परिपत्र|आदेश|शासन निर्णय|अधिसूचना)",
    r"उपरोक्त (दस्तावेज़|दस्तावेज|परिपत्र|आदेश)",
    r"इसमें (बताया|दिया|उल्लेख)",
]

def has_self_reference(text):
    if not text:
        return False
    return any(re.search(p, text, re.IGNORECASE) for p in SELF_REFERENCE_PATTERNS)


def filter_questions(questions):
    """Drop questions that leak self-reference to 'this document' etc.
    Officers query cold with no file open, so these are unanswerable in isolation
    and will fail eval for reasons unrelated to actual retrieval/generation quality."""
    clean, dropped = [], 0
    for q in questions:
        if has_self_reference(q.get("query", "")) or has_self_reference(q.get("expected_answer", "")):
            dropped += 1
            continue
        clean.append(q)
    if dropped:
        print(f"  [FILTER] Dropped {dropped} self-referential question(s)")
    return clean


def filter_unverified_terms(questions, doc_text):
    """Drop expected_terms that aren't literal substrings of the document text.
    The prompt asks for verbatim terms for exact-match scoring, but the model
    sometimes writes a description instead (e.g. 'Date is March 7, 2019' rather
    than the actual date string as it appears in the doc). A term that can never
    match will silently fail every future eval run regardless of assistant quality."""
    dropped_terms = 0
    for q in questions:
        terms = q.get("expected_terms", [])
        kept = [t for t in terms if t and t in doc_text]
        dropped_terms += len(terms) - len(kept)
        q["expected_terms"] = kept
    if dropped_terms:
        print(f"  [FILTER] Dropped {dropped_terms} non-verbatim expected_term(s)")
    return questions


VERIFY_PROMPT = """You are fact-checking a benchmark test set generated for a document QA system. Below is a source document and a list of question/answer pairs that are supposed to be answerable from it.

For EACH pair, check two things:
1. Does the answer actually address the question that was asked (not a different fact from the document)?
2. Is the answer factually supported by the document text, without invented details?

Return ONLY a JSON array, same length and order as the input pairs, no markdown, no explanation:
[
  {{"valid": true}},
  {{"valid": false, "reason": "answer addresses a different question / not supported by document"}},
  ...
]

QUESTION/ANSWER PAIRS TO CHECK:
{pairs_json}

SOURCE DOCUMENT:
---
{doc_text}
---"""


def verify_questions(gemini_client, questions, doc_text, filename):
    """Second-pass check: does each generated answer actually answer its own
    question, and is it grounded in the document? Catches cross-contamination
    where the model answers question A with a fact that belongs to question B,
    which a plain substring check on expected_terms won't reliably catch."""
    if not questions:
        return questions

    pairs = [{"query": q.get("query", ""), "expected_answer": q.get("expected_answer", "")} for q in questions]
    prompt = (
        VERIFY_PROMPT
        .replace("{pairs_json}", json.dumps(pairs, ensure_ascii=False, indent=2))
        .replace("{doc_text}", doc_text)
    )

    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.0,
                response_mime_type="application/json",
            )
        )
        text = response.text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        verdicts = json.loads(text)
    except Exception as e:
        print(f"  [WARN] Verification pass failed for {filename}, keeping questions unverified: {e}")
        return questions

    if len(verdicts) != len(questions):
        print(f"  [WARN] Verification returned {len(verdicts)} verdicts for {len(questions)} questions, "
              f"skipping verification for {filename}")
        return questions

    clean, dropped = [], 0
    for q, v in zip(questions, verdicts):
        if v.get("valid", False):
            clean.append(q)
        else:
            dropped += 1
            reason = v.get("reason", "unspecified")
            print(f"  [VERIFY] Dropped: \"{q.get('query', '')[:70]}...\" ({reason})")
    if dropped:
        print(f"  [VERIFY] Dropped {dropped} question(s) that failed fact-check")
    return clean


def read_document(filepath):
    """Read document text - PDF via pymupdf4llm, txt directly."""
    if filepath.endswith(".pdf") or filepath.endswith(".PDF"):
        try:
            return pymupdf4llm.to_markdown(filepath)
        except Exception as e:
            print(f"  [WARN] PDF read failed: {e}")
            return None
    else:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            return f.read()


def build_prompt(n_questions, extra_requirements, doc_text):
    """Substitute placeholders manually (not .format()) so stray { } in
    government document text (tables, embedded codes, scanned artifacts)
    can't raise a KeyError and kill the run."""
    return (
        GENERATION_PROMPT
        .replace("{n_questions}", str(n_questions))
        .replace("{extra_requirements}", extra_requirements)
        .replace("{doc_text}", doc_text)
    )


def generate_questions_for_doc(gemini_client, doc_text, filename, n_questions=3, trim_chars=9000):
    """Call Gemini to generate questions for a single document."""
    if len(doc_text) > trim_chars:
        doc_text = doc_text[:trim_chars] + "\n\n[... document continues, truncated ...]"

    extra = ""
    if n_questions >= 5:
        extra = ("5. One GR number lookup question, phrased naturally, e.g. asking for the "
                 "GR number tied to a named scheme, college, or department action - never "
                 "\"the GR number of this document.\"")

    prompt = build_prompt(n_questions, extra, doc_text)

    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.3,
                response_mime_type="application/json",
            )
        )
        text = response.text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        questions = json.loads(text)
        questions = filter_questions(questions)
        questions = filter_unverified_terms(questions, doc_text)
        questions = verify_questions(gemini_client, questions, doc_text, filename)
        for q in questions:
            # assigned in code, not left for the model to fill in and possibly hallucinate
            q["source_doc"] = filename
        return questions
    except Exception as e:
        print(f"  [ERROR] Generation failed for {filename}: {e}")
        return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Max documents to scan (cap on corpus pass, not on questions)")
    parser.add_argument("--target", type=int, default=100, help="Target total question count; keeps pulling from remaining docs until reached or corpus exhausted")
    parser.add_argument("--out", default="benchmark/benchmark_100.json", help="Output file")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output file")
    parser.add_argument(
        "--max-doc-chars", type=int, default=15000,
        help="Skip docs larger than this many characters (default 15000 ~3750 tokens). "
             "Use 0 to disable skipping."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for shuffling document order, so the sampled set isn't just "
             "an alphabetical/directory-listing prefix. Fixed default keeps runs reproducible; "
             "pass a different value to get a different random sample."
    )
    args = parser.parse_args()

    gemini = get_genai_client()

    # Dedupe: on case-insensitive filesystems (macOS/Windows), *.pdf and *.PDF
    # match the same files, so concatenating both lists doubles every PDF.
    pdf_files = sorted(set(glob.glob("docs/*.pdf") + glob.glob("docs/*.PDF")))
    orgpedia_en = sorted(glob.glob("docs/parsed/*.en.txt"))

    all_docs = [(f, "pdf") for f in pdf_files] + [(f, "orgpedia") for f in orgpedia_en]

    # Shuffle BEFORE limiting, otherwise --limit just takes an alphabetical prefix
    # of the directory listing, which can silently overrepresent whatever department/
    # year happens to sort first.
    rng = random.Random(args.seed)
    rng.shuffle(all_docs)

    if args.limit:
        all_docs = all_docs[:args.limit]

    print(f"Total documents available: {len(all_docs)} (PDFs: {len(pdf_files)}, Orgpedia: {len(orgpedia_en)})")
    print(f"Target question count: {args.target}")

    already_done = set()
    all_questions = []
    if args.resume and os.path.exists(args.out):
        with open(args.out, "r", encoding="utf-8") as f:
            all_questions = json.load(f)
        already_done = {q.get("source_doc") for q in all_questions if q.get("source_doc")}
        print(f"Resuming: {len(already_done)} docs already processed, {len(all_questions)} questions loaded")

    if not any(q.get("category") == "not_found" for q in all_questions):
        all_questions.extend(NOT_FOUND_QUESTIONS)
        print(f"Added {len(NOT_FOUND_QUESTIONS)} not-found questions")

    generated_count = len(all_questions)
    docs_skipped_size = 0
    docs_failed = 0

    for i, (filepath, doc_type) in enumerate(all_docs):
        if generated_count >= args.target:
            print(f"\nTarget of {args.target} reached, stopping scan early.")
            break

        filename = os.path.basename(filepath)
        if filename in already_done:
            print(f"[{i+1}/{len(all_docs)}] SKIP {filename} (already done)")
            continue

        try:
            if doc_type == "pdf":
                doc_text = pymupdf4llm.to_markdown(filepath)
            else:
                with open(filepath, "r", encoding="utf-8") as f:
                    doc_text = f.read()
        except Exception as e:
            print(f"[{i+1}/{len(all_docs)}] SKIP {filename} (Error reading file: {e})")
            docs_failed += 1
            continue

        print(f"[{i+1}/{len(all_docs)}] Processing {filename} ({len(doc_text):,} chars)...")

        max_chars = args.max_doc_chars
        if max_chars > 0 and len(doc_text) > max_chars:
            print(f"  [SKIP] Too large ({len(doc_text):,} chars > {max_chars:,} limit). "
                  f"Run with --max-doc-chars 0 to force truncation instead.")
            docs_skipped_size += 1
            continue

        n_q = 5 if doc_type == "pdf" else 4
        questions = generate_questions_for_doc(gemini, doc_text, filename, n_q, trim_chars=9000)

        if questions:
            print(f"  Generated {len(questions)} questions (after filtering)")
            all_questions.extend(questions)
            generated_count = len(all_questions)

            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(all_questions, f, indent=2, ensure_ascii=False)
        else:
            docs_failed += 1

        time.sleep(0.5)

    print(f"\nDone! Total questions: {len(all_questions)}")
    if generated_count < args.target:
        print(f"[NOTE] Fell short of target ({generated_count}/{args.target}). "
              f"{docs_skipped_size} docs skipped for size, {docs_failed} failed generation/read. "
              f"Widen the corpus (raise --limit), or raise --max-doc-chars, to close the gap.")
    print(f"Saved to: {args.out}")

    cats = {}
    for q in all_questions:
        c = q.get("category", "unknown")
        cats[c] = cats.get(c, 0) + 1
    print("\nBy category:")
    for cat, count in sorted(cats.items()):
        print(f"  {cat}: {count}")


if __name__ == "__main__":
    main()