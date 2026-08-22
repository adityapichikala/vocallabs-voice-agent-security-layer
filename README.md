# Black Box for Voice Agents

**A real-time guardrail layer for AI voice calls.**

## 1. User Persona
**Priya, QA Lead at a BPO.** Priya manages a team of human agents and is now overseeing a fleet of AI voice agents. Her biggest nightmare is an AI agent hallucinating a policy or promising a refund that the company cannot honor. She needs a real-time monitor to catch these errors before the call ends.

## 2. Prior Art Check
Unlike post-call QA tools (like Cogito or Cresta) which analyze call transcripts hours or days after the call has ended, **Black Box operates in REAL-TIME**. It uses a dual-LLM scoring pipeline and Server-Sent Events (SSE) to flag hallucinations, broken promises, and low confidence *while the call is still happening*, allowing for immediate human escalation.

## 3. Cost Ceiling
We maintain a strict budget of **< $0.50 per 100 calls**. This is achieved by utilizing:
- **Groq Whisper** for lightning-fast, low-cost transcription.
- **Gemini 1.5 Flash** as our primary scorer (extremely cheap and fast).
- **Groq Llama 3** (free tier/low cost) and local **Ollama** as zero-cost fallbacks.

## 4. Demo Walkthrough Script (5 minutes)
- **Step 1:** Load the Monitor Dashboard and select **"Test 1 — Clean Call"**. Hit Analyze to show baseline performance (all green).
- **Step 2:** Select **"Test 2 — Hallucination"**. Show the dashboard instantly catching the agent promising a policy-violating refund, turning the banner RED.
- **Step 3:** Toggle the **"🔥 Simulate API Outage"** button. Select **"Test 4 — Low Confidence"**. 
- **Step 4:** Hit Analyze and point out the **Provider Badge** changing from `gemini` to `groq` to prove the fallback cascade works live.
- **Step 5:** Navigate to the **Evaluation Dashboard (`/eval`)** to show our 83.3% test harness score and honest failure log.

## 5. Setup Instructions
1. Clone the repository and navigate to the project root.
2. Create a `.env` file and add your `GEMINI_API_KEY` and `GROQ_API_KEY`.
3. Start the FastAPI backend:
   ```bash
   uvicorn main:app --reload --port 8000
   ```
4. Start the Next.js frontend:
   ```bash
   cd frontend
   npm run dev
   ```
5. Open `http://localhost:3000` in your browser.
