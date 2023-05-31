from setuptools import setup, find_packages

with open('README.md') as f:
    LONG_DESCRIPTION = f.read()

setup(
    name = 'ipdw',
    version = '1.0.0',
    license = 'MIT',
    description = 'Inverse-Path-Distance-Weighted Interpolation for Python',
    long_description=LONG_DESCRIPTION,
    long_description_content_type = 'text/markdown',
    author = 'Kyle Wright',
    author_email = 'Kyle.Wright@twdb.texas.gov',
    url = 'https://github.com/wrightky/ipdw',
    packages = find_packages(),
    keywords='interpolation ipdw idw',
    classifiers = [
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.11',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: GIS'
    ],
    install_requires = ['numpy','matplotlib','scikit-fmm'],
)
