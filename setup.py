from setuptools import setup, find_packages


setup(
    name="stock-ana",
    version="0.0.1",
    description="A stock analysis framework",
    author="Rossi",
    packages=find_packages(exclude=("test", "test.*", "data")),
    include_package_data=True,
    data_files=[
        ("templates", ["templates/pdf.html"])
    ]
)
