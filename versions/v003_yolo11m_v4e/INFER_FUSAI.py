"""Version-local batch V4E inference; use --check to inspect without inference."""
from pathlib import Path
import sys

if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parent / 'inference_tools'))
    from batch_fusai import main
    main()
