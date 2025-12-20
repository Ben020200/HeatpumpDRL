from setuptools import setup, find_packages
from pathlib import Path

this_dir = Path(__file__).parent
readme = (this_dir / "README.md").read_text(encoding="utf-8")

setup(
    name="heatpump-rl-env",
    version="0.1.0",
    author="Your Name",
    description="Thermal environment for heat pump RL testing",
    long_description=readme,
    long_description_content_type="text/markdown",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "gymnasium>=0.28.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "pandas>=1.3.0",
        "pyyaml>=5.4",
    ],
)