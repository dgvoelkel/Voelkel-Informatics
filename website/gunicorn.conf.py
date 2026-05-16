import os

workers = 1
timeout = 120
bind = f"0.0.0.0:{os.getenv('PORT', '10000')}"
