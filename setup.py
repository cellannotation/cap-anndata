from setuptools import setup, find_packages

setup(
    name='cap_anndata',
    version='0.1.0',
    author='R. Mukhin, A. Isaev',
    author_email='roman@ebookapplications.com',
    packages=find_packages(exclude=["test"]),
    description='Partial read of AnnData files for low-memory operations with large datasets.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/cellannotation/cap-anndata',
    project_urls={
        "Bug Tracker": "https://github.com/cellannotation/cap-anndata/issues"
    },
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
    install_requires=[
        "numpy>=1.26.3",
        "pandas>=2.2.0",
        "anndata>=0.10.5"
    ],
    extras_require={"dev": ["pytest>=8.0.0"]}
)
