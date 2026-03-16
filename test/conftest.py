# Test configuration for TFMPE package
import os

# Use platform allocator so GPU OOM errors immediately instead of
# retrying indefinitely via the BFC allocator.
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")