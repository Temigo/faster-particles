from distutils.core import setup

setup(
    name='faster-particles',
    version='0.1.0',
    author='Laura Domine, Ji Won Park, Kazuhiro Terao',
    author_email='temigo@gmx.com',
    packages=['faster_particles', 'faster_particles.test', 'faster_particles.toydata', 'faster_particles.larcvdata'],
    entry_points={
        'console_scripts': [
            'ppn=bin.ppn:main'
        ],
    },
    license='LICENSE.md',
    description='Point Proposal Network for particles images and related tools.',
    long_description=open('README.md').read(),
    url='https://github.com/Temigo/faster-particles',
    install_requires=[
        "matplotlib >= 1.5.3",
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
