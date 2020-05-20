import setuptools

setuptools.setup(
    name='gmw',
    version='0.0.1',
    author='Matteo Dora',
    author_email='matteo.dora@ens.psl.eu',
    description='Generalised Morse Wavelets',
    url='https://github.com/mattbit/gmw',
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    install_requires=[
        'numpy',
        'scipy',
    ]
)
