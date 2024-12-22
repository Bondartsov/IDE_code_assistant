# conftest.py

import warnings

warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    module=r"numpy\..*"
)

warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    module=r"faiss\..*"
)