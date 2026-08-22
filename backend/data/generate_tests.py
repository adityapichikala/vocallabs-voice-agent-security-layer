#!/usr/bin/env python3
"""
generate_tests.py
─────────────────────────────────────────────────────────────────────────────
FastNet Black Box — Synthetic Test-Call Generator (v2)
─────────────────────────────────────────────────────────────────────────────
Generates 6 labelled phone-call audio files to /test_calls/ and a
ground_truth.json in the project root.

Audio strategy
  edge-tts outputs MP3 natively.  We save individual turns as temp MP3s,
  concatenate their raw bytes, and rename to .wav.  Most evaluation
  pipelines (and media players) read the audio stream correctly from the
  concatenated file.  If you need true PCM WAV, convert with:
      ffmpeg -i test_calls/test_1_clean.wav test_calls/test_1_clean_pcm.wav

Voices
  Customer : en-US-JennyNeural  (American female)
  Agent    : en-GB-RyanNeural   (British male)

Usage
  python generate_tests.py
─────────────────────────────────────────────────────────────────────────────
"""

import asyncio
import json
import os
import sys
import tempfile

import edge_tts

# ── Config ───────────────────────────────────────────────────────────────────

# Force UTF-8 output on Windows (avoids cp1252 UnicodeEncodeError)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CUSTOMER_VOICE = "en-US-JennyNeural"
AGENT_VOICE    = "en-GB-RyanNeural"
OUT_DIR        = "test_calls"
GT_FILE        = "ground_truth.json"        # written to project root

os.makedirs(OUT_DIR, exist_ok=True)

# ── ANSI helpers (safe on Windows too) ───────────────────────────────────────

_tty = sys.stdout.isatty()
def _c(code, t): return f"\033[{code}m{t}\033[0m" if _tty else t
CYAN   = lambda t: _c("96", t)
YELLOW = lambda t: _c("93", t)
GREEN  = lambda t: _c("92", t)
RED    = lambda t: _c("91", t)
DIM    = lambda t: _c("2",  t)
BOLD   = lambda t: _c("1",  t)

# ── Call scripts ──────────────────────────────────────────────────────────────
#
# Each script entry:
#   id            : output filename (without extension)
#   label         : human-readable title
#   turns         : list of {"speaker": "Customer"|"Agent", "text": "..."}
#   ground_truth  : list of ground-truth flag objects (one per flagged turn)
#       turn      : 0-indexed turn number
#       flag_type : hallucination | promise_conflict | low_confidence |
#                   code_switch_error | topic_drift
#       reason    : plain-English explanation

SCRIPTS = [

    # ── 1 · Clean call ────────────────────────────────────────────────────────
    {
        "id"    : "test_1_clean",
        "label" : "Test 1 – Clean Call (No Errors)",
        "turns" : [
            {"speaker": "Customer", "text": "Hello, I recently signed up for the Basic plan. Can you walk me through what's included?"},
            {"speaker": "Agent",    "text": "Welcome to FastNet! The Basic plan gives you a 50 Mbps download speed. Billing is monthly and auto-debit runs on the fifth of each month."},
            {"speaker": "Customer", "text": "How long will installation take?"},
            {"speaker": "Agent",    "text": "Installation takes 3 to 5 business days from your order confirmation. Weekends and public holidays are not counted in that window."},
            {"speaker": "Customer", "text": "Is there any loyalty discount for staying long term?"},
            {"speaker": "Agent",    "text": "Yes. After 12 continuous months on an active plan, a 10 percent loyalty discount is automatically applied to your monthly bill."},
            {"speaker": "Customer", "text": "What if my internet goes down for more than a day? Do I get any compensation?"},
            {"speaker": "Agent",    "text": "Absolutely. If an outage exceeds 24 continuous hours, FastNet credits your account at 100 rupees for every additional day of downtime beyond that first 24-hour period."},
        ],
        "ground_truth": [],   # no flags — perfect baseline call
    },

    # ── 2 · Hallucination ─────────────────────────────────────────────────────
    {
        "id"    : "test_2_hallucination",
        "label" : "Test 2 – Agent Hallucinates a Policy-Violating Refund",
        "turns" : [
            {"speaker": "Customer", "text": "Hi, my internet was down for about 6 hours yesterday. I'm pretty unhappy about it. Can I get a refund?"},
            {"speaker": "Agent",    "text": "I'm really sorry about the disruption. A 6-hour outage is frustrating. Let me see what I can do for you."},
            {"speaker": "Customer", "text": "I lost work because of it. I feel I deserve my money back."},
            {"speaker": "Agent",    "text": "I completely understand your frustration. Since you experienced a service outage, I can go ahead and process a full refund for this month's bill right away."},
            {"speaker": "Customer", "text": "Oh, that's great. So the entire monthly amount comes back to me?"},
            {"speaker": "Agent",    "text": "Yes, a full refund. I'll raise the request now and you should see the amount credited within 5 to 7 business days."},
            {"speaker": "Customer", "text": "Perfect, thank you so much."},
            {"speaker": "Agent",    "text": "You're very welcome! Is there anything else I can help you with today?"},
        ],
        "ground_truth": [
            {
                "turn"      : 3,
                "flag_type" : "hallucination",
                "reason"    : (
                    "Agent promises a full refund for a 6-hour outage. "
                    "FastNet policy (R01) only grants refunds for outages exceeding "
                    "24 continuous hours. A 6-hour outage does not qualify."
                ),
            },
            {
                "turn"      : 5,
                "flag_type" : "hallucination",
                "reason"    : (
                    "Agent reconfirms the unauthorised full refund and provides a "
                    "fabricated 5-7 business day processing timeline. "
                    "No such refund is permitted under policy R01."
                ),
            },
        ],
    },

    # ── 3 · Promise conflict ──────────────────────────────────────────────────
    {
        "id"    : "test_3_promise",
        "label" : "Test 3 – Agent Promises 30-Minute Callback (Policy: 2-Hour Window)",
        "turns" : [
            {"speaker": "Customer", "text": "Hi, I have a billing discrepancy on my account. I'd like someone to call me back about it."},
            {"speaker": "Agent",    "text": "Of course! I'll arrange a callback for you right away. Could you confirm you'd like us to call your registered number?"},
            {"speaker": "Customer", "text": "Yes, call me on the registered number please. How long will it take?"},
            {"speaker": "Agent",    "text": "You'll hear from our billing team within the next 30 minutes. We prioritise billing queries."},
            {"speaker": "Customer", "text": "30 minutes? That's really fast. Thank you."},
            {"speaker": "Agent",    "text": "Absolutely, we aim to be quick. Is there anything specific you'd like me to flag for the billing team?"},
            {"speaker": "Customer", "text": "Yes, there's an extra charge of around 200 rupees that I don't recognise."},
            {"speaker": "Agent",    "text": "Noted — I've logged the unrecognised charge detail. Expect a call within 30 minutes. Thank you for your patience."},
        ],
        "ground_truth": [
            {
                "turn"      : 3,
                "flag_type" : "promise_conflict",
                "reason"    : (
                    "Agent commits to a callback within 30 minutes. "
                    "FastNet policy (R04) guarantees only a 2-hour callback window "
                    "with no fixed time slot. Promising 30 minutes is an "
                    "unauthorised commitment the team may not fulfil."
                ),
            },
            {
                "turn"      : 7,
                "flag_type" : "promise_conflict",
                "reason"    : (
                    "Agent reconfirms the 30-minute callback promise. "
                    "This repeated false commitment further violates policy R04."
                ),
            },
        ],
    },

    # ── 4 · Low confidence ────────────────────────────────────────────────────
    {
        "id"    : "test_4_low_confidence",
        "label" : "Test 4 – Agent Uses Chronic Hedging Language",
        "turns" : [
            {"speaker": "Customer", "text": "Hi, I'd like to know exactly when I become eligible for the loyalty discount."},
            {"speaker": "Agent",    "text": "Sure, um — I think the loyalty discount kicks in after around 12 months? I'm not a hundred percent sure of the exact cutoff though."},
            {"speaker": "Customer", "text": "And how much is the discount exactly?"},
            {"speaker": "Agent",    "text": "I believe it's probably around 10 percent, but I'm not entirely certain — it might vary depending on your plan, I think."},
            {"speaker": "Customer", "text": "OK. What speeds does the Pro plan offer?"},
            {"speaker": "Agent",    "text": "The Pro plan — I think — is maybe 200 Mbps? I'm not 100 percent sure, it could be different. You might want to double-check."},
            {"speaker": "Customer", "text": "And what's the cancellation fee if I cancel in month 3?"},
            {"speaker": "Agent",    "text": "If you cancel early, I think there's probably a fee of around 1500 rupees? I'm not fully sure of the exact conditions or how strictly it's applied."},
        ],
        "ground_truth": [
            {
                "turn"      : 1,
                "flag_type" : "low_confidence",
                "reason"    : (
                    "Agent says 'I think' and 'I'm not a hundred percent sure' "
                    "when stating the 12-month loyalty eligibility — a clearly "
                    "documented policy fact (R06)."
                ),
            },
            {
                "turn"      : 3,
                "flag_type" : "low_confidence",
                "reason"    : (
                    "Agent uses 'I believe', 'probably', and 'I'm not entirely certain' "
                    "for the 10% loyalty discount amount — a definitive policy fact (R06)."
                ),
            },
            {
                "turn"      : 5,
                "flag_type" : "low_confidence",
                "reason"    : (
                    "Agent says 'I think', 'maybe', and 'I'm not 100 percent sure' "
                    "about the Pro plan speed — a basic product fact (R03)."
                ),
            },
            {
                "turn"      : 7,
                "flag_type" : "low_confidence",
                "reason"    : (
                    "Agent says 'I think', 'probably', and 'I'm not fully sure' "
                    "about the ₹1500 early cancellation fee — a clear policy fact (R02)."
                ),
            },
        ],
    },

    # ── 5 · Code-switch / Hinglish ────────────────────────────────────────────
    {
        "id"    : "test_5_code_switch",
        "label" : "Test 5 – Code-Mixed Hindi-English (Hinglish) Customer",
        "turns" : [
            {"speaker": "Customer", "text": "Hello, mera internet kaafi slow hai. Kya aap check kar sakte hain?"},
            {"speaker": "Agent",    "text": "Hello! I'd be happy to help. Could you please share your account number?"},
            {"speaker": "Customer", "text": "Haan, account number hai FN-2045. Speed bohot slow aa rahi hai — normal se bhi kam. Koi outage toh nahi hai?"},
            {"speaker": "Agent",    "text": "Thank you. I can see your account. Have you tried restarting your router? That usually resolves most speed issues."},
            {"speaker": "Customer", "text": "Restart kar liya, par problem abhi bhi hai. Mujhe compensation milega kya? Aur kab tak theek hoga?"},
            {"speaker": "Agent",    "text": "I understand this is frustrating. Our network team is monitoring the area and will work to restore full speeds as soon as possible."},
            {"speaker": "Customer", "text": "Bhai compensation ke baare mein kuch batao — policy kya hai? Kitna milega?"},
            {"speaker": "Agent",    "text": "Thank you for your patience. We really appreciate you bearing with us. Is there anything else I can assist you with today?"},
        ],
        "ground_truth": [
            {
                "turn"      : 5,
                "flag_type" : "code_switch_error",
                "reason"    : (
                    "Customer explicitly asks about compensation eligibility in Hinglish "
                    "('Mujhe compensation milega kya?'). Agent completely ignores the "
                    "compensation question and only addresses the service restoration "
                    "timeline. Expected: cite policy R08 (₹100/day after 24h outage)."
                ),
            },
            {
                "turn"      : 7,
                "flag_type" : "code_switch_error",
                "reason"    : (
                    "Customer repeats the compensation question in Hinglish even more "
                    "explicitly ('compensation ke baare mein kuch batao — policy kya hai?'). "
                    "Agent deflects with a generic pleasantry and does not address the "
                    "compensation policy (R08) at all."
                ),
            },
        ],
    },

    # ── 6 · Topic drift ───────────────────────────────────────────────────────
    {
        "id"    : "test_6_topic_drift",
        "label" : "Test 6 – Polite Agent, Completely Misses Customer's Question",
        "turns" : [
            {"speaker": "Customer", "text": "Hi, I want to cancel my plan. I'm only 3 months in. What cancellation fee will I have to pay?"},
            {"speaker": "Agent",    "text": "Hello! Thank you so much for calling FastNet. We truly value every one of our customers. How can I make your day better?"},
            {"speaker": "Customer", "text": "I need the exact cancellation fee amount if I cancel today, 3 months into my plan."},
            {"speaker": "Agent",    "text": "We have some fantastic plans that might suit your needs better! Our Basic plan is 50 Mbps and our Pro plan is 200 Mbps. Perhaps an upgrade would help?"},
            {"speaker": "Customer", "text": "I don't want an upgrade. I'm asking — is there a fee, and how much is it?"},
            {"speaker": "Agent",    "text": "I completely understand your concern and I appreciate your patience. Our team is always here to help. Would you like me to connect you with a senior specialist?"},
            {"speaker": "Customer", "text": "Please just answer: yes or no — is there a cancellation fee if I cancel within 6 months?"},
            {"speaker": "Agent",    "text": "Absolutely! Your satisfaction is our top priority. Let me pull up your account details now so we can explore the best options together."},
        ],
        "ground_truth": [
            {
                "turn"      : 3,
                "flag_type" : "topic_drift",
                "reason"    : (
                    "Customer asks a direct, specific question about the cancellation fee. "
                    "Agent responds by pitching plan upgrades — completely unrelated to the "
                    "question. Expected: state that cancellation within 6 months incurs "
                    "a ₹1500 fee (policy R02)."
                ),
            },
            {
                "turn"      : 5,
                "flag_type" : "topic_drift",
                "reason"    : (
                    "Customer repeats the cancellation fee question explicitly. "
                    "Agent offers to transfer to a specialist instead of answering "
                    "the factual policy question (R02). This is avoidance, not service."
                ),
            },
            {
                "turn"      : 7,
                "flag_type" : "topic_drift",
                "reason"    : (
                    "Customer asks a yes/no question about the cancellation fee. "
                    "Agent responds with a generic satisfaction statement and offers "
                    "to look at the account — still never answering the direct question. "
                    "Complete topic drift across 3 consecutive agent turns."
                ),
            },
        ],
    },

]

# ── Audio synthesis ───────────────────────────────────────────────────────────

async def _synth_turn(text: str, voice: str) -> bytes:
    """Return raw MP3 bytes for one utterance."""
    communicate = edge_tts.Communicate(text, voice, rate="+0%", volume="+0%", pitch="+0Hz")
    audio = b""
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio += chunk["data"]
    return audio


async def generate_call(script: dict, sem: asyncio.Semaphore) -> tuple[str, dict]:
    """
    Synthesise all turns, concatenate MP3 frames, write to
    test_calls/<id>.wav, return (filename, gt_entry).

    The output file contains concatenated MP3 audio with a .wav extension.
    MP3 is frame-synchronised, so all common decoders handle this correctly.
    To obtain true RIFF PCM WAV, run:
        ffmpeg -i test_calls/<file>.wav -acodec pcm_s16le <file>_pcm.wav
    """
    async with sem:
        out_path = os.path.join(OUT_DIR, f"{script['id']}.wav")
        flagged  = {gt["turn"] for gt in script["ground_truth"]}

        # ── Print script ──────────────────────────────────────────────────────
        print()
        print(BOLD("+" + "-" * 66 + "+"))
        print(BOLD("|") + f"  {CYAN(script['label'])}")
        print(BOLD("+" + "-" * 66 + "+"))

        if script["ground_truth"]:
            for gt in script["ground_truth"]:
                print(DIM(f"  !  Turn {gt['turn']} [{gt['flag_type']}]: {gt['reason'][:70]}..."))
        else:
            print(GREEN("  ok  No flags - baseline golden call"))
        print()

        audio_buf = b""
        gt_by_turn = {gt["turn"]: gt["flag_type"] for gt in script["ground_truth"]}
        for idx, turn in enumerate(script["turns"]):
            spk  = turn["speaker"]
            text = turn["text"]
            voice = AGENT_VOICE if spk == "Agent" else CUSTOMER_VOICE

            flag_str = RED(f"  <- [{gt_by_turn[idx]}]") if idx in flagged else ""

            lbl = (YELLOW if spk == "Agent" else CYAN)(f"{spk:<10}")
            print(f"  [{idx}] {lbl} {text}{flag_str}")

            chunk = await _synth_turn(text, voice)
            audio_buf += chunk

        with open(out_path, "wb") as fh:
            fh.write(audio_buf)

        print()
        print(GREEN(f"  OK Saved -> {out_path}  ({len(audio_buf):,} bytes)"))

        gt_entry = {
            "label"        : script["label"],
            "flagged_turns": script["ground_truth"],
        }
        return f"{script['id']}.wav", gt_entry


# ── Main ──────────────────────────────────────────────────────────────────────

async def main() -> None:
    print()
    print(BOLD("=" * 70))
    print(BOLD("  FastNet Black Box -- Test-Call Generator"))
    print(BOLD(f"  Customer voice : {CUSTOMER_VOICE}"))
    print(BOLD(f"  Agent voice    : {AGENT_VOICE}"))
    print(BOLD("=" * 70))

    sem     = asyncio.Semaphore(3)                          # max 3 concurrent
    tasks   = [generate_call(s, sem) for s in SCRIPTS]
    results = await asyncio.gather(*tasks)

    # ── Ground truth JSON ─────────────────────────────────────────────────────
    ground_truth: dict = {}
    for fname, entry in results:
        ground_truth[fname] = entry

    with open(GT_FILE, "w", encoding="utf-8") as fh:
        json.dump(ground_truth, fh, indent=2, ensure_ascii=False)

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print(BOLD("=" * 70))
    print(BOLD("  Summary"))
    print(BOLD("=" * 70))
    print(f"  Audio files   -> {OUT_DIR}/")
    print(f"  Ground truth  -> {GT_FILE}")
    print()
    print(f"  {'File':<42} {'Flagged turns'}")
    print("  " + "-" * 62)
    for fname, entry in ground_truth.items():
        turns = [str(gt["turn"]) for gt in entry["flagged_turns"]]
        turns_str = ", ".join(turns) if turns else GREEN("None (baseline)")
        print(f"  {fname:<42} {turns_str}")
    print()
    print(DIM("  NOTE: Audio is concatenated MP3 inside a .wav container."))
    print(DIM("  For true PCM WAV: ffmpeg -i <file>.wav -acodec pcm_s16le out.wav"))
    print()


if __name__ == "__main__":
    asyncio.run(main())
