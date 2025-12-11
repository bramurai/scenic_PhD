#!/bin/bash
# Download just the simple_tokenizer from CLIP (without installing the full package)

set -e

echo "Installing minimal CLIP tokenizer for dmvr..."

# Get the site-packages directory
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
CLIP_DIR="$SITE_PACKAGES/clip"

echo "Creating clip package in: $CLIP_DIR"

# Create clip package directory
mkdir -p "$CLIP_DIR"

# Download simple_tokenizer.py
echo "Downloading simple_tokenizer.py..."
curl -s https://raw.githubusercontent.com/openai/CLIP/main/clip/simple_tokenizer.py \
    -o "$CLIP_DIR/simple_tokenizer.py"

# Download bpe_simple_vocab_16e6.txt.gz (required by tokenizer)
echo "Downloading vocabulary file..."
curl -s https://github.com/openai/CLIP/raw/main/clip/bpe_simple_vocab_16e6.txt.gz \
    -o "$CLIP_DIR/bpe_simple_vocab_16e6.txt.gz"

# Create __init__.py to make it a package
cat > "$CLIP_DIR/__init__.py" << 'EOF'
"""Minimal CLIP package with just simple_tokenizer for dmvr."""
from .simple_tokenizer import SimpleTokenizer
__all__ = ['SimpleTokenizer', 'simple_tokenizer']
EOF

echo "✓ CLIP tokenizer installed successfully!"
echo "  Location: $CLIP_DIR"
ls -lh "$CLIP_DIR"
