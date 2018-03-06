from distutils.core import setup

setup(
    name='faster-particles',
    version='0.1.0',
    author='Laura Domine and Ji Won Park',
    author_email='temigo@gmx.com',
    packages=['faster_particles', 'faster_particles.test', 'faster_particles.toydata'],
    scripts=['bin/ppn-demo','bin/ppn-train'],
    url='http://pypi.python.org/pypi/faster-particles/',
    license='LICENSE.md',
    description='Useful faster-particles-related stuff.',
    long_description=open('README.md').read(),
    install_requires=[
        "matplotlib == 1.5.3",
        "numpy == 1.13.1",
        "scikit-image == 0.12.3",
        "tensorflow == 1.3.1"
    ],
)
