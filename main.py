#!/usr/bin/env python3
"""
Example run script for Neural Digit package.

Usage examples:

    # Train on MNIST with default settings
    python mai.py --mode train --dataset mnist --epochs 50

    # Train with specific ablation mode
    python main.py --mode train --dataset mnist --ablation baseline_cnn

    # Resume training
    python main.py --mode train --checkpoint save_file_name/run_name.last.pt

    # Evaluate a checkpoint
    python main.py --mode eval --checkpoint save_file_name/ocr_hybrid.best.pt

    # Export to ONNX
    python main.py --mode export --checkpoint save_file_name/ocr_hybrid.best.pt

    # Single image prediction
    python main.py --mode predict --checkpoint save_file_name/ocr_hybrid.best.pt --image test.png

    # Run corruption benchmark
    python main.py --mode bench --checkpoint save_file_name/ocr_hybrid.best.pt

    # Train on EMNIST balanced
    python main.py --mode train --dataset emnist --emnist-split balanced --epochs 100
"""

import sys
from NeuralDigit.cli import main    #Change this "from" import if your main function is located in a different module/file

if __name__ == "__main__":
    sys.exit(main())
