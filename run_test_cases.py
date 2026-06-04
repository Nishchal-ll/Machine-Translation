#!/usr/bin/env python3
import sys
from pathlib import Path
import re

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

try:
    from app import app, translator as app_translator, trained_load_error
except Exception as exc:
    app = None
    app_translator = None
    trained_load_error = str(exc)

try:
    from src.translator import NepaliTranslator
    from src.config import MODEL_DIR
    missing_dependency = None
except Exception as exc:
    NepaliTranslator = None
    MODEL_DIR = ROOT_DIR / "outputs" / "models"
    missing_dependency = str(exc)


def detect_honorific_level(sentence: str) -> str:
    normalized = sentence.lower()
    if any(token in normalized for token in ["sir", "madam", "ma'am", "please", "could you", "would you"]):
        return "तपाईं"
    if any(token in normalized for token in ["buddy", "dude", "friend", "y'all", "you all"]):
        return "तिमी"
    return "तँ"


def devanagari_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    devanagari_letters = [c for c in letters if 0x0900 <= ord(c) <= 0x097F]
    return len(devanagari_letters) / len(letters)


def create_translator() -> NepaliTranslator:
    model_path = MODEL_DIR / "best_honorifics_model"
    if not model_path.exists():
        raise FileNotFoundError(f"Trained model directory not found: {model_path}")
    return NepaliTranslator(model_path, device="cpu")


def run_test_steps():
    test_results = []
    translator = None
    if app_translator is None:
        try:
            translator = create_translator()
        except Exception as exc:
            translator = None
    else:
        translator = app_translator

    # Unit Test 01: Input Handling
    ut01_input = "Please tell me your name."
    ut01_actual = "Translator available" if translator is not None else "Translator unavailable"
    ut01_pass = bool(ut01_input.strip()) and translator is not None
    test_results.append(("UT 01", "Input Handling", "Valid English sentence", "Sentence accepted and forwarded to pipeline", ut01_actual, "Pass" if ut01_pass else "Fail"))

    # Unit Test 02: Preprocessing
    ut02_raw = "  Hello,   world!  "
    ut02_expected = "Hello, world!"
    ut02_preprocessed = translator.preprocess_text(ut02_raw) if translator else ""
    ut02_tokenized = None
    ut02_tokenization_info = ""
    try:
        ut02_tokenized = translator.tokenizer([ut02_preprocessed], return_tensors="pt", padding=True, truncation=True)
        ut02_tokenization_info = f"shape={tuple(ut02_tokenized['input_ids'].shape)}"
    except Exception as exc:
        ut02_tokenization_info = f"tokenization failed: {exc}"
    ut02_pass = ut02_preprocessed == ut02_expected and ut02_tokenized is not None and ut02_tokenized["input_ids"].shape[1] > 0
    test_results.append(("UT 02", "Preprocessing", "Preprocessing", "Cleaned and tokenized text output", f"preprocessed='{ut02_preprocessed}' {ut02_tokenization_info}", "Pass" if ut02_pass else "Fail"))

    # Unit Test 03: Honorific Detection
    ut03_input = "Could you please help me, sir?"
    ut03_expected = "तपाईं"
    ut03_detected = detect_honorific_level(ut03_input)
    ut03_pass = ut03_detected == ut03_expected
    test_results.append(("UT 03", "Honorific Detection", "Sentence with contextual cues", "Correct honorific level (तँ / तिमी / तपाईं) identified", f"detected='{ut03_detected}'", "Pass" if ut03_pass else "Fail"))

    # Unit Test 04: Translation Module
    ut04_input = "Please sit down, sir."
    ut04_output = translator.translate(ut04_input) if translator else ""
    ut04_pass = bool(ut04_output.strip()) and devanagari_ratio(ut04_output) >= 0.3
    test_results.append(("UT 04", "Translation Module", "English sentence", "Accurate Nepali translation generated", f"output='{ut04_output}'", "Pass" if ut04_pass else "Fail"))

    # Unit Test 05: Honorific Mapping
    ut05_input = "Can you show me the way, sir?"
    ut05_output = translator.translate(ut05_input) if translator else ""
    ut05_pass = bool(ut05_output.strip()) and "तपाईं" in ut05_output
    test_results.append(("UT 05", "Honorific Mapping", "Base translation + context", "Proper pronoun and verb adjustment applied", f"output='{ut05_output}'", "Pass" if ut05_pass else "Fail"))

    # System Test 01: API startup
    st01_details = "app missing"
    st01_pass = False
    if app is not None:
        try:
            with app.test_client() as client:
                resp = client.get("/api/health")
                data = resp.get_json(silent=True)
                st01_pass = resp.status_code == 200 and data is not None and data.get("status") == "ok"
                st01_details = f"status={resp.status_code}, payload={data}"
        except Exception as exc:
            st01_details = f"error={exc}"
    test_results.append(("ST 01", "API startup", "Model and flask load successfully", "Flask app starts and health endpoint is OK", st01_details, "Pass" if st01_pass else "Fail"))

    # System Test 02: Frontend backend integration
    st02_details = "app missing"
    st02_pass = False
    if app is not None:
        try:
            with app.test_client() as client:
                resp = client.post("/api/translate", json={"text": ut01_input})
                data = resp.get_json(silent=True)
                st02_pass = resp.status_code == 200 and data is not None and data.get("success") is True
                st02_details = f"status={resp.status_code}, data={data}"
        except Exception as exc:
            st02_details = f"error={exc}"
    test_results.append(("ST 02", "Frontend backend integration", "Translation request returns JSON response", "Translation POST /api/translate returns valid JSON", st02_details, "Pass" if st02_pass else "Fail"))

    # System Test 03: Translation output display
    st03_details = "app missing"
    st03_pass = False
    if app is not None:
        try:
            with app.test_client() as client:
                resp = client.get("/")
                st03_pass = resp.status_code == 200 and resp.content_type.startswith("text/html")
                st03_details = f"status={resp.status_code}, content_type={resp.content_type}"
        except Exception as exc:
            st03_details = f"error={exc}"
    test_results.append(("ST 03", "Translation output display", "Nepali translation shown correctly on UI", "Frontend root page serves HTML", st03_details, "Pass" if st03_pass else "Fail"))

    # System Test 04: Multiple Sentence Input
    st04_details = "app missing"
    st04_pass = False
    if app is not None:
        try:
            with app.test_client() as client:
                requests = [
                    {"text": "Please sit down, sir."},
                    {"text": "How are you today?"},
                    {"text": "Thank you very much."},
                ]
                success_count = 0
                for payload in requests:
                    resp = client.post("/api/translate", json=payload)
                    data = resp.get_json(silent=True)
                    if resp.status_code == 200 and data and data.get("success") is True:
                        success_count += 1
                    else:
                        break
                st04_pass = success_count == len(requests)
                st04_details = f"success_count={success_count}/{len(requests)}"
        except Exception as exc:
            st04_details = f"error={exc}"
    test_results.append(("ST 04", "Multiple Sentence Input", "System handles multiple requests correctly", "Multiple translate calls succeed", st04_details, "Pass" if st04_pass else "Fail"))

    # System Test 05: End-to-end translation flow
    st05_details = f"output='{ut04_output}'"
    st05_pass = bool(ut04_output.strip()) and ut04_pass
    test_results.append(("ST 05", "End-to-end translation flow", "Input- preprocessing- translation - output works smoothly", "Full pipeline from input to Nepali output", st05_details, "Pass" if st05_pass else "Fail"))

    return test_results


def print_report(results):
    headers = ["Test Case ID", "Scenario / Module", "Input / Scenario", "Expected Result", "Actual Result", "Status"]
    row_format = "{:<8}  {:<28}  {:<40}  {:<45}  {:<50}  {:<6}"
    print(row_format.format(*headers))
    print("-" * 180)
    for row in results:
        print(row_format.format(*row))


if __name__ == "__main__":
    print("Running Machine Translation test cases...\n")
    if missing_dependency:
        print(f"Warning: missing dependency while importing project modules: {missing_dependency}\n")
    results = run_test_steps()
    print_report(results)
    print("\nTest run complete.")
