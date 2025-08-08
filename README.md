# YouTube Transcript Summarizer & Q&A

A simple Gradio app that fetches a YouTube video transcript and lets you:
- Summarize the video
- Ask questions about its content

Backend has been refactored to use OpenAI (via LangChain) for LLM + embeddings.

## Setup
1) Python 3.10+
2) Install dependencies:
```
python -m pip install -r requirements.txt
```
3) Create a `.env` file with your OpenAI key:
```
OPENAI_API_KEY=sk-...  # already present in this workspace
# optional
OPENAI_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

## Optional: transcript access
- Public videos often work without cookies. YouTube may block cloud IPs.
- If needed, add `youtube_cookies.txt` (Netscape format) to the project root. The app will automatically load only the essential cookies.
- Proxies: set `YT_HTTP_PROXY` and `YT_HTTPS_PROXY` if you must use a residential proxy. Proxies are ignored when cookies are present.
- Manual fallback: paste a transcript in the "Paste transcript (optional)" box to bypass fetching entirely.

## Run
```
python ytbot.py
# Open http://localhost:7860
```

## Notes
- Logic and UI preserved from the original one-file app.
- If transcript fetch fails in a cloud environment, run locally or use cookies/residential proxy.
