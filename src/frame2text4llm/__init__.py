from .package_metadata import (
    __doc__,
    __name__,
    __author__,
    __email__,
    __version__,
    __url__,
)
from beartype.claw import beartype_this_package

beartype_this_package()
