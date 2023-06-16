class UnsupportedDimensionError(Exception):
    """Raised when a mesh with an unsupported dimension is passed to a function."""
    def __init__(self, _supports_only: list[str], message = None):
        if not self.message:
            self.message = f"Unsupported dimension. Only {_supports_only} meshes are supported."
        super().__init__(self.message)