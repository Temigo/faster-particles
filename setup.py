#from distutils.core import setup
from setuptools import setup, find_packages

setup(
    name='faster-particles',
    version='0.2.1',
    author='Laura Domine, Ji Won Park, Kazuhiro Terao',
    author_email='temigo@gmx.com',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'ppn=faster_particles.bin.ppn:main'
        ],
    },
    license='LICENSE.md',
    description='Point Proposal Network for particles images and related tools.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/Temigo/faster-particles',
    install_requires=[
        "matplotlib >= 2.2.2",
        "numpy >= 1.13.1",
        "scikit-learn >= 0.18.1",
        "scikit-image >= 0.12.3",
        "tensorflow >= 1.3.1"
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2.7',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Recognition'
    ],
    keywords='physics',
    project_urls={
        'Bug Reports': 'https://github.com/Temigo/faster-particles/issues',
        'Source': 'https://github.com/Temigo/faster-particles',
    },
)
