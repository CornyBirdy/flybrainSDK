"""``python -m flybrain.malecns download`` - build the connectome cache."""

import sys

from .dataset import main

if __name__ == "__main__":
    sys.exit(main())
