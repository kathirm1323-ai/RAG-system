import os

# Bind dynamically to the port provided by Render, falling back to 10000
port = os.environ.get("PORT", "10000")
bind = f"0.0.0.0:{port}"

# Limit workers to 1 to optimize memory usage (prevent OOM) on the 512MB free tier
workers = 1
threads = 4
timeout = 120
