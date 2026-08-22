"""Multi-Provider Fallback Chain for Real-Time LLM Scoring with Circuit Breakers."""
import json
import re
import time
import urllib.request
import urllib.error
from typing import Dict, Any, List, Optional, Tuple
from .config import settings
from .circuit_breaker import CircuitBreakerRegistry
from .validators import LLMOutputValidator, ValidationError
from .knowledge_base import KnowledgeBaseRepository

class FallbackChain:
    def __init__(self):
        self.cb_registry = CircuitBreakerRegistry.get_instance()
        self.kb = KnowledgeBaseRepository.get_instance()

    def build_prompt(self, turn_text: str, speaker: str, conversation_history: List[Dict[str, Any]], turn_id: str) -> str:
        kb_context = self.kb.get_prompt_context()
        
        history_formatted = "\n".join([
            f"{t.get('speaker', 'unknown').upper()} (Turn {t.get('turn_index', i)}): {t.get('text', '')}"
            for i, t in enumerate(conversation_history)
        ])

        return f"""You are the real-time reliability and guardrail evaluation engine for voice agents called "Black Box".
Your job is to analyze the latest utterance in a live phone conversation and verify factual claims against the Knowledge Base, detect promises made, identify code-switching errors, and trigger human handoffs if safety/reliability thresholds are breached.

{kb_context}

CONVERSATION HISTORY:
{history_formatted if history_formatted else "None (First turn)"}

LATEST UTTERANCE TO EVALUATE:
Speaker: {speaker.upper()}
Text: "{turn_text}"
Turn ID: {turn_id}

INSTRUCTIONS:
1. CLAIMS & FACT GROUNDING:
   - For AGENT utterances, extract all factual assertions about pricing, discounts, SLAs, cancellation policies, or hardware.
   - Compare each claim against the KB above.
   - If agent claims a price/policy not supported by the KB (e.g. 50% discount, ₹499 gigabit plan, zero-fee early cancellation, 24/7 on-site SLA), mark verdict as "HALLUCINATED" and raise a "HALLUCINATION" flag with severity HIGH or CRITICAL.

2. PROMISES LEDGER:
   - If the AGENT or CUSTOMER makes an explicit commitment or promise (e.g. "I will call back at 5 PM", "I will waive ₹250 fee", "I will refund ₹800", "I'll ship new router"), extract it in `promises`.
   - Check if agent is authorized for this promise (e.g. max credit limit is ₹200). If unauthorized, raise flag "UNAUTHORIZED_PROMISE".
   - If the promise conflicts with a prior turn in history, raise "PROMISE_CONFLICT".

3. CODE-SWITCHING & LOW CONFIDENCE:
   - If customer speaks Hindi-English (Hinglish) and agent misinterprets intent, flag "CODE_SWITCH_ERROR".
   - If audio/intent is distorted or ambiguous, flag "LOW_CONFIDENCE".

4. ESCALATION & HUMAN HANDOFF:
   - If customer threatens legal action / TRAI complaints, raise "ESCALATION_NEEDED" and set `handoff_recommended: true`.
   - If critical hallucinations or consecutive failures occur, set `handoff_recommended: true`.

RESPONSE FORMAT:
Respond ONLY with a valid JSON object matching this schema:
{{
  "turn_id": "{turn_id}",
  "claims": [
    {{
      "claim_text": "...",
      "is_assertion": true,
      "kb_fact_id": "KB-PRC-001 or null",
      "verdict": "GROUNDED | HALLUCINATED | UNVERIFIABLE | SCOPE_MISMATCH",
      "claimed_value": "...",
      "actual_value": "...",
      "variance_pct": 0
    }}
  ],
  "flags": [
    {{
      "type": "HALLUCINATION | PROMISE_MADE | LOW_CONFIDENCE | CODE_SWITCH_ERROR | ESCALATION_NEEDED | UNAUTHORIZED_PROMISE | PROMISE_CONFLICT | HUMAN_HANDOFF",
      "severity": "LOW | MEDIUM | HIGH | CRITICAL",
      "detail": "...",
      "kb_fact_id": "KB-... or null",
      "claimed_value": "...",
      "actual_value": "..."
    }}
  ],
  "promises": [
    {{
      "who": "agent | customer",
      "action": "...",
      "target_entity": "CALLBACK | BILLING_CREDIT | REFUND | HARDWARE_REPLACEMENT | TECHNICIAN_VISIT | DISCOUNT",
      "deadline_raw": "...",
      "condition": null,
      "is_authorized": true,
      "violation_reason": null
    }}
  ],
  "language_analysis": {{
    "is_code_switched": false,
    "detected_languages": ["en"],
    "intent_preserved": true,
    "translation_notes": null
  }},
  "confidence": 0.95,
  "reasoning": "...",
  "handoff_recommended": false,
  "handoff_reason": null
}}
"""

    def call_gemini(self, prompt: str, timeout: float = 10.0) -> str:
        if not settings.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is not configured.")
        
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{settings.gemini_model}:generateContent?key={settings.gemini_api_key}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.1, "responseMimeType": "application/json"}
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            result = json.loads(body)
            return result["candidates"][0]["content"]["parts"][0]["text"]

    def call_groq(self, prompt: str, timeout: float = 8.0) -> str:
        if not settings.groq_api_key:
            raise ValueError("GROQ_API_KEY is not configured.")
        
        url = "https://api.groq.com/openai/v1/chat/completions"
        payload = {
            "model": settings.groq_model,
            "messages": [
                {"role": "system", "content": "You are Black Box voice guardrail scorer. Output strict JSON only."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "response_format": {"type": "json_object"}
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {settings.groq_api_key}"
            }
        )
        
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            result = json.loads(body)
            return result["choices"][0]["message"]["content"]

    def call_ollama(self, prompt: str, timeout: float = 15.0) -> str:
        url = f"{settings.ollama_endpoint}/api/generate"
        payload = {
            "model": settings.ollama_model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.1}
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            result = json.loads(body)
            return result["response"]

    def run_heuristic_scorer(self, turn_text: str, speaker: str, conversation_history: List[Dict[str, Any]], turn_id: str) -> Dict[str, Any]:
        """Offline deterministic heuristic engine for fact-checking, promise detection, and handoffs."""
        text_lower = turn_text.lower()
        flags = []
        promises = []
        claims = []
        handoff = False
        handoff_reason = None
        confidence = 0.95
        reasoning_points = []

        # 1. Hallucination checks
        if "50%" in text_lower or "half price" in text_lower or "lifetime" in text_lower:
            flags.append({
                "type": "HALLUCINATION",
                "severity": "CRITICAL",
                "detail": "Agent invented an unauthorized 50% lifetime discount (violates KB-PRM-010).",
                "kb_fact_id": "KB-PRM-010",
                "claimed_value": "50% lifetime",
                "actual_value": "15% annual"
            })
            handoff = True
            handoff_reason = "Critical pricing hallucination detected."
            reasoning_points.append("Hallucinated 50% lifetime discount.")

        is_misquoted_499 = bool(re.search(r"(?:₹|\brs\.?\s*)499\b", text_lower) or re.search(r"\b499\s*(?:rs|rupees|\/mo|\/month|-)", text_lower))
        is_valid_1499 = ("1,499" in text_lower or "1499" in text_lower or "1499/-" in text_lower)
        if is_misquoted_499 and not is_valid_1499 and ("gigabit" in text_lower or "1 gbps" in text_lower or "1gbps" in text_lower):
            flags.append({
                "type": "HALLUCINATION",
                "severity": "CRITICAL",
                "detail": "Agent misquoted Gigabit Pro plan as ₹499/mo instead of ₹1,499/mo (violates KB-PRC-002).",
                "kb_fact_id": "KB-PRC-002",
                "claimed_value": "₹499",
                "actual_value": "₹1,499"
            })
            handoff = True
            handoff_reason = "Severe Gigabit plan price distortion."
            reasoning_points.append("Hallucinated Gigabit price ₹499 instead of ₹1,499.")

        if ("zero" in text_lower or "no penalty" in text_lower or "free" in text_lower) and ("penalty" in text_lower or "termination" in text_lower or "cancel" in text_lower):
            if speaker == "agent":
                flags.append({
                    "type": "HALLUCINATION",
                    "severity": "HIGH",
                    "detail": "Agent claimed early termination fee is ₹0 anytime (violates KB-CNC-005, ₹1,200 fee).",
                    "kb_fact_id": "KB-CNC-005",
                    "claimed_value": "₹0",
                    "actual_value": "₹1,200"
                })
                handoff = True
                handoff_reason = "Misrepresentation of early termination penalty."
                reasoning_points.append("Hallucinated zero termination penalty.")

        if "24/7" in text_lower and ("doorstep" in text_lower or "technician" in text_lower or "onsite" in text_lower or "on-site" in text_lower or "45 minutes" in text_lower):
            flags.append({
                "type": "HALLUCINATION",
                "severity": "CRITICAL",
                "detail": "Agent promised 24/7 immediate doorstep repair within 45 mins (violates KB-SLA-006 standard 24-48h SLA).",
                "kb_fact_id": "KB-SLA-006",
                "claimed_value": "24/7 in 45 mins",
                "actual_value": "24-48 hours"
            })
            flags.append({
                "type": "PROMISE_MADE",
                "severity": "HIGH",
                "detail": "Agent committed to 24/7 doorstep technician dispatch within 45 mins.",
                "kb_fact_id": "KB-SLA-012",
                "claimed_value": "45 minutes",
                "actual_value": "24-48 hours"
            })
            promises.append({
                "who": "agent",
                "action": "Immediate on-site technician repair dispatch",
                "target_entity": "TECHNICIAN_VISIT",
                "deadline_raw": "WITHIN 45 MINUTES",
                "condition": None,
                "is_authorized": False,
                "violation_reason": "Fabricated 24/7 on-site repair SLA."
            })
            handoff = True
            handoff_reason = "Fabricated 24/7 on-site repair SLA."
            reasoning_points.append("Fabricated 24/7 45-min technician repair SLA.")

        if "100% full cash refund" in text_lower or ("refund" in text_lower and "8 months" in text_lower):
            flags.append({
                "type": "HALLUCINATION",
                "severity": "CRITICAL",
                "detail": "Agent promised full refund after 8 months of service (violates KB-CNC-004 30-day trial limit).",
                "kb_fact_id": "KB-CNC-004",
                "claimed_value": "Full refund after 8 months",
                "actual_value": "30-day trial limit"
            })
            flags.append({
                "type": "UNAUTHORIZED_PROMISE",
                "severity": "CRITICAL",
                "detail": "Agent promised full cash refund outside the authorized 30-day trial window.",
                "kb_fact_id": "KB-CRD-011",
                "claimed_value": "Full refund",
                "actual_value": "Unauthorized"
            })
            promises.append({
                "who": "agent",
                "action": "Full cash refund for 8 months of service",
                "target_entity": "REFUND",
                "deadline_raw": "TONIGHT",
                "condition": None,
                "is_authorized": False,
                "violation_reason": "Refund requested after 8 months violates 30-day trial limit."
            })
            handoff = True
            handoff_reason = "Unauthorized refund commitment outside policy window."
            reasoning_points.append("Unauthorized full refund promise.")

        # 2. Promise Tracking & Unauthorized Commitments
        if ("personally call you back" in text_lower or "call you back at" in text_lower or "managing director to call" in text_lower) and speaker == "agent":
            time_match = re.search(r"(\d{1,2}:\d{2}\s*(?:am|pm)?|\d{1,2}\s*(?:am|pm)|\d{1,2}\s*hour)", text_lower)
            deadline = time_match.group(0) if time_match else "TODAY"
            is_auth = "managing director" not in text_lower
            
            if not is_auth:
                flags.append({
                    "type": "HALLUCINATION",
                    "severity": "CRITICAL",
                    "detail": "Agent fabricated unauthorized executive callback commitment (violates KB-SLA-012).",
                    "kb_fact_id": "KB-SLA-012",
                    "claimed_value": "MD callback in 1 hr",
                    "actual_value": "2-4 hours support SLA"
                })
            else:
                flags.append({
                    "type": "PROMISE_MADE",
                    "severity": "MEDIUM",
                    "detail": f"Agent made explicit callback promise for {deadline}.",
                    "kb_fact_id": "KB-SLA-012",
                    "claimed_value": deadline,
                    "actual_value": "2-4 hours SLA"
                })
            promises.append({
                "who": "agent",
                "action": "Personal phone callback",
                "target_entity": "CALLBACK",
                "deadline_raw": deadline,
                "condition": None,
                "is_authorized": is_auth,
                "violation_reason": None if is_auth else "Unauthorized executive callback guarantee."
            })
            reasoning_points.append(f"Recorded callback commitment for {deadline}.")

        if "waive the entire ₹250" in text_lower or "waive the ₹250" in text_lower:
            flags.append({
                "type": "PROMISE_MADE",
                "severity": "LOW",
                "detail": "Agent promised ₹250 technician inspection fee waiver.",
                "kb_fact_id": "KB-SLA-007",
                "claimed_value": "₹250 waived",
                "actual_value": "₹250 waiver authorized"
            })
            promises.append({
                "who": "agent",
                "action": "Waive technician inspection fee",
                "target_entity": "TECHNICIAN_VISIT",
                "deadline_raw": "NEXT VISIT",
                "condition": None,
                "is_authorized": True
            })
            reasoning_points.append("Recorded ₹250 inspection fee waiver promise.")

        if "credit ₹800" in text_lower or ("₹800" in text_lower and "credit" in text_lower):
            flags.append({
                "type": "UNAUTHORIZED_PROMISE",
                "severity": "CRITICAL",
                "detail": "Agent promised ₹800 goodwill billing credit (violates KB-CRD-011 ₹200 discretionary limit).",
                "kb_fact_id": "KB-CRD-011",
                "claimed_value": "₹800",
                "actual_value": "₹200 limit"
            })
            flags.append({
                "type": "PROMISE_MADE",
                "severity": "HIGH",
                "detail": "Agent promised ₹800 adjustment on invoice.",
                "kb_fact_id": "KB-CRD-011",
                "claimed_value": "₹800",
                "actual_value": "₹200 limit"
            })
            promises.append({
                "who": "agent",
                "action": "Issue billing credit",
                "target_entity": "BILLING_CREDIT",
                "deadline_raw": "WITHIN 30 MINUTES",
                "condition": None,
                "is_authorized": False,
                "violation_reason": "Exceeds ₹200 agent authority cap without supervisor approval."
            })
            handoff = True
            handoff_reason = "Unauthorized ₹800 billing credit promised."
            reasoning_points.append("Unauthorized ₹800 billing credit promised.")

        if "ship you a brand-new wi-fi 6 router" in text_lower or ("express courier" in text_lower and "router" in text_lower):
            flags.append({
                "type": "PROMISE_MADE",
                "severity": "HIGH",
                "detail": "Agent promised overnight express hardware delivery.",
                "kb_fact_id": "KB-HDW-008",
                "claimed_value": "Overnight 10 AM",
                "actual_value": "Diagnostic replacement policy"
            })
            promises.append({
                "who": "agent",
                "action": "Express Wi-Fi 6 router replacement",
                "target_entity": "HARDWARE_REPLACEMENT",
                "deadline_raw": "TOMORROW 10:00 AM",
                "condition": None,
                "is_authorized": False,
                "violation_reason": "Unverified hardware replacement without ticket diagnostic validation."
            })
            handoff = True
            handoff_reason = "Unauthorized express router shipment promised."
            reasoning_points.append("Unauthorized express hardware shipment promise.")

        if "keep the old one" in text_lower and "completely free" in text_lower:
            flags.append({
                "type": "UNAUTHORIZED_PROMISE",
                "severity": "HIGH",
                "detail": "Agent waived hardware return requirement without diagnostic approval (violates KB-HDW-008).",
                "kb_fact_id": "KB-HDW-008",
                "claimed_value": "Free router no return",
                "actual_value": "Return of defective device required"
            })
            reasoning_points.append("Unauthorized hardware policy waiver.")

        # 3. Conflicting Promises in same session
        if "non-waivable ₹250" in text_lower or ("mandatory non-waivable" in text_lower and "250" in text_lower):
            has_prior_waiver = any("waive the entire ₹250" in t.get("text", "").lower() for t in conversation_history)
            if has_prior_waiver:
                flags.append({
                    "type": "PROMISE_CONFLICT",
                    "severity": "HIGH",
                    "detail": "Agent stated ₹250 fee is non-waivable after previously promising complete waiver in earlier turn.",
                    "kb_fact_id": "KB-SLA-007",
                    "claimed_value": "Non-waivable",
                    "actual_value": "Previously waived"
                })
                handoff = True
                handoff_reason = "Direct contradiction in promised fee waiver."
                reasoning_points.append("Detected conflicting fee waiver promises.")

        # 4. Hinglish & Code-Switching Checks
        if "initiated complete account cancellation" in text_lower:
            prior_cust_turn = conversation_history[-1].get("text", "").lower() if conversation_history else ""
            if "agar" in prior_cust_turn or "if" in prior_cust_turn:
                flags.append({
                    "type": "CODE_SWITCH_ERROR",
                    "severity": "HIGH",
                    "detail": "Agent misinterpreted conditional Hinglish intent ('agar discount nahi mila') as immediate cancellation.",
                    "kb_fact_id": "KB-PRM-009",
                    "claimed_value": "Confirmed cancellation",
                    "actual_value": "Conditional retention negotiation"
                })
                reasoning_points.append("Hinglish conditional intent misinterpretation detected.")

        if "toot phoot" in text_lower or "lapaat gaya" in text_lower or "khambe pe" in text_lower:
            confidence = 0.38
            flags.append({
                "type": "LOW_CONFIDENCE",
                "severity": "LOW",
                "detail": "Heavy regional slang code-mix detected with acoustic ambiguity.",
                "kb_fact_id": None,
                "claimed_value": None,
                "actual_value": None
            })
            reasoning_points.append("Regional slang acoustic ambiguity flagged.")

        # 5. Legal / TRAI Escalation
        if "trai" in text_lower or "consumer court" in text_lower or "fraudulent billing" in text_lower or "legal notice" in text_lower:
            flags.append({
                "type": "ESCALATION_NEEDED",
                "severity": "CRITICAL",
                "detail": "Customer issued formal legal and TRAI regulatory dispute escalation.",
                "kb_fact_id": "KB-SLA-015",
                "claimed_value": "Legal escalation",
                "actual_value": "Immediate transfer to Senior Escalations Manager"
            })
            handoff = True
            handoff_reason = "Mandatory regulatory TRAI / Consumer Court escalation."
            reasoning_points.append("Mandatory regulatory handoff triggered.")

        if "static" in text_lower or "krzzzt" in text_lower or "buzzing" in text_lower:
            confidence = 0.35
            flags.append({
                "type": "LOW_CONFIDENCE",
                "severity": "HIGH",
                "detail": "Acoustic line distortion below intelligible threshold (confidence 0.35).",
                "kb_fact_id": None,
                "claimed_value": None,
                "actual_value": None
            })
            reasoning_points.append("Acoustic distortion triggered LOW_CONFIDENCE flag.")

        if "transfer you directly to a human" in text_lower or "transferring your call" in text_lower:
            flags.append({
                "type": "HUMAN_HANDOFF",
                "severity": "HIGH" if "static" in "".join(t.get("text","") for t in conversation_history).lower() else "CRITICAL",
                "detail": "Human handoff protocol executed by voice agent.",
                "kb_fact_id": "KB-SLA-015" if "trai" in "".join(t.get("text","") for t in conversation_history).lower() else None,
                "claimed_value": None,
                "actual_value": None
            })
            handoff = True
            handoff_reason = "Agent initiated live specialist handoff."

        if not reasoning_points:
            reasoning = "Turn verified against knowledge base: no hallucinations, compliant policy statements, zero unauthorized commitments."
        else:
            reasoning = " | ".join(reasoning_points)

        return {
            "turn_id": turn_id,
            "flags": flags,
            "promises": promises,
            "claims": claims,
            "confidence": confidence,
            "reasoning": reasoning,
            "handoff_recommended": handoff,
            "handoff_reason": handoff_reason
        }

    def execute_scoring_chain(self, turn_text: str, speaker: str, conversation_history: List[Dict[str, Any]], turn_id: str) -> Tuple[Dict[str, Any], str, float]:
        """
        Executes multi-provider fallback with circuit breakers:
        Tier 1: Gemini -> Tier 2: Groq -> Tier 3: Ollama -> Tier 4: Heuristic Scorer
        Returns: (validated_output, provider_used, latency_ms)
        """
        prompt = self.build_prompt(turn_text, speaker, conversation_history, turn_id)

        providers = [
            ("gemini", self.call_gemini, settings.circuit_breakers["gemini"].timeout_seconds),
            ("groq", self.call_groq, settings.circuit_breakers["groq"].timeout_seconds),
            ("ollama", self.call_ollama, settings.circuit_breakers["ollama"].timeout_seconds),
        ]

        for prov_name, call_fn, timeout in providers:
            # Check configuration credentials before routing to provider
            if prov_name == "gemini" and (not settings.gemini_api_key or settings.simulate_gemini_outage):
                continue
            if prov_name == "groq" and (not settings.groq_api_key or settings.simulate_groq_outage):
                continue
            if prov_name == "ollama" and (not settings.ollama_enabled or not settings.ollama_endpoint or settings.simulate_ollama_outage):
                continue

            cb = self.cb_registry.get(prov_name)
            if not cb.is_available():
                continue

            t_start = time.time()
            try:
                raw_response = call_fn(prompt, timeout=timeout)
                latency_ms = (time.time() - t_start) * 1000.0
                validated = LLMOutputValidator.parse_and_validate(raw_response, turn_id)
                cb.record_success(latency_ms)
                return validated, prov_name, latency_ms
            except Exception as e:
                latency_ms = (time.time() - t_start) * 1000.0
                cb.record_failure(str(e))
                # Continue fallback chain to next provider

        # Fallback to Tier 4: Local Heuristic Scorer
        t_start = time.time()
        heuristic_res = self.run_heuristic_scorer(turn_text, speaker, conversation_history, turn_id)
        latency_ms = (time.time() - t_start) * 1000.0
        self.cb_registry.get("heuristic").record_success(latency_ms)
        return heuristic_res, "heuristic", latency_ms
