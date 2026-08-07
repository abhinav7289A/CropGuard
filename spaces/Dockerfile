# Space image for the demo panel. Docker rather than the Streamlit SDK because the Hub no
# longer accepts `streamlit` as an SDK for new Spaces — only gradio, docker and static.
#
# Separate from the repo-root Dockerfile, which builds the FastAPI serving image. That one
# bakes in weights and serves an API; this one runs a UI and fetches weights at runtime.
FROM python:3.11-slim

# Spaces run the container as uid 1000. Installing as root and running as user leaves pip's
# cache unwritable, so the user is created first and everything happens inside its home.
RUN useradd -m -u 1000 user
USER user
ENV PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    PYTHONUNBUFFERED=1

WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .

# 7860 is the Spaces default and is declared as `app_port` in the Space card.
EXPOSE 7860

# fileWatcherType=none because nothing edits the source at runtime, and the watcher costs CPU
# that the free tier would rather spend on inference.
#
# XSRF protection is off, and that is required rather than lazy. Spaces serves the app inside
# an iframe on huggingface.co, so Streamlit's XSRF cookie is a third-party cookie and modern
# browsers drop it. The token then never reaches /_stcore/upload_file, which rejects every
# upload with 403 - the file uploader is simply broken with it on. CORS goes with it because
# Streamlit ignores enableCORS=false while XSRF is enabled, so the pair has to move together.
#
# What that costs: a third-party page could POST a file to this app. There is no auth, no
# stored state and no cross-user data here - one anonymous visitor cannot reach another's
# session - so the exposure is an unwanted inference, not a leak.
CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.port=7860", "--server.address=0.0.0.0", \
     "--server.headless=true", "--server.fileWatcherType=none", \
     "--server.enableXsrfProtection=false", "--server.enableCORS=false", \
     "--server.maxUploadSize=10", \
     "--browser.gatherUsageStats=false"]
