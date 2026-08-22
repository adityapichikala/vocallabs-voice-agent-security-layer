"""Pre-flight validation script to verify Knowledge Base, Test Cases, and Ground-Truth integrity."""
import json
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BACKEND_DIR / "data"
KB_PATH = DATA_DIR / "knowledge_base.json"
TEST_CASES_PATH = DATA_DIR / "test_cases_metadata.json"

def validate_all():
    print("=" * 60)
    print(" BLACK BOX PRE-FLIGHT DATASET & KNOWLEDGE BASE VALIDATOR")
    print("=" * 60 + "\n")

    errors = []
    warnings = []

    # 1. Validate Knowledge Base
    print("1. Checking Knowledge Base JSON...")
    if not KB_PATH.exists():
        errors.append(f"Knowledge Base file missing at {KB_PATH}")
        return errors, warnings

    with open(KB_PATH, "r", encoding="utf-8") as f:
        kb_data = json.load(f)

    facts = kb_data.get("facts", [])
    print(f"   [OK] Loaded {len(facts)} facts from KB version {kb_data.get('version')}.")
    
    known_kb_ids = set()
    for fact in facts:
        f_id = fact.get("id")
        if not f_id:
            errors.append("Found fact without 'id'.")
        elif f_id in known_kb_ids:
            errors.append(f"Duplicate fact ID '{f_id}'.")
        known_kb_ids.add(f_id)

    # 2. Validate Test Cases
    print("\n2. Checking 20 Benchmark Test Cases...")
    if not TEST_CASES_PATH.exists():
        errors.append(f"Test cases file missing at {TEST_CASES_PATH}")
        return errors, warnings

    with open(TEST_CASES_PATH, "r", encoding="utf-8") as f:
        tc_data = json.load(f)

    test_cases = tc_data.get("test_cases", [])
    print(f"   [OK] Loaded {len(test_cases)} test cases from benchmark suite.")

    if len(test_cases) != 20:
        warnings.append(f"Expected 20 test cases, found {len(test_cases)}.")

    categories_count = {}
    for tc in test_cases:
        tc_id = tc.get("id")
        cat = tc.get("category", "UNKNOWN")
        categories_count[cat] = categories_count.get(cat, 0) + 1

        # Check dialogue turns
        dialogue = tc.get("dialogue", [])
        if not dialogue:
            errors.append(f"Test case {tc_id} has empty dialogue.")

        last_time = -0.01
        for turn in dialogue:
            s_time = turn.get("start_time", 0.0)
            e_time = turn.get("end_time", 0.0)
            if s_time < last_time:
                errors.append(f"{tc_id} turn {turn.get('turn_index')} has non-monotonic start time ({s_time} < {last_time}).")
            if e_time < s_time:
                errors.append(f"{tc_id} turn {turn.get('turn_index')} has end_time < start_time.")
            last_time = s_time

        # Check ground truth
        gt = tc.get("expected_ground_truth", {})
        for ef in gt.get("expected_flags", []):
            kb_id = ef.get("kb_fact_id")
            if kb_id and kb_id not in known_kb_ids:
                errors.append(f"{tc_id} references non-existent KB fact ID: '{kb_id}'")

    print("\n3. Category Distribution Balance:")
    for cat, count in categories_count.items():
        print(f"   - {cat}: {count} cases")

    print("\n" + "=" * 60)
    if errors:
        print(f"[FAIL] Pre-flight validation found {len(errors)} ERRORS:")
        for err in errors:
            print(f"  ❌ {err}")
        return False
    else:
        print("[SUCCESS] All Knowledge Base facts and 20 Test Cases are 100% VALID!")
        if warnings:
            print(f"\n[WARNINGS] ({len(warnings)}):")
            for w in warnings:
                print(f"  ⚠️  {w}")
        return True

if __name__ == "__main__":
    success = validate_all()
    sys.exit(0 if success else 1)
