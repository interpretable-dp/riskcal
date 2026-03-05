from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import os

NO_COMPILE = os.environ.get("RISKCAL_NO_COMPILE") == "1"


class OptionalBuildExt(build_ext):
    """Build Cython extensions, silently falling back to pure Python on failure."""

    def run(self):
        if NO_COMPILE:
            print(
                "riskcal: RISKCAL_NO_COMPILE=1 -> skipping Cython compilation; "
                "using pure Python fallback."
            )
            return
        try:
            super().run()
        except Exception as e:
            print("riskcal: Cython compilation failed; falling back to pure Python.")
            print(f"  {e}")

    def build_extension(self, ext):
        try:
            super().build_extension(ext)
            print(f"riskcal: Cython compiled successfully: {ext.name}")
        except Exception as e:
            print(
                f"riskcal: WARNING: Cython compilation failed; "
                f"using pure Python for {ext.name}"
            )
            print(f"  {e}")


def get_extensions():
    if NO_COMPILE:
        return []

    try:
        from Cython.Build import cythonize
        import numpy as np
    except Exception as e:
        print("riskcal: Cython/numpy not available for build; using pure Python.")
        print(f"  {e}")
        return []

    ext = Extension(
        name="riskcal.analysis.rdp.converter",
        sources=["src/riskcal/analysis/rdp/converter.pyx"],
        include_dirs=[np.get_include()],
    )

    return cythonize(
        [ext],
        compiler_directives={"language_level": "3"},
        annotate=False,
    )


setup(
    package_dir={"": "src"},
    ext_modules=get_extensions(),
    cmdclass={"build_ext": OptionalBuildExt},
)
