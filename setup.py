from setuptools import setup, find_packages

setup(
    name='cap_anndata',
    version='0.1.0',
    author='R. Mukhin, A. Isaev',
    author_email='roman@ebookapplications.com',
    packages=find_packages(),
    description='A brief description of your package',
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
)
