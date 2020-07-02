import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="medcam",
    version="0.1.04",
    author="Karol Gotkowski",
    author_email="KarolGotkowski@gmx.de",
    description="An easy to use library that makes model predictions more interpretable for humans.",
    long_description="An easy to use library that makes model predictions more interpretable for humans. M3d-CAM allows the generation of attention maps with multiple methods like Guided Backpropagation, Grad-Cam, Guided Grad-Cam and Grad-Cam++.",
    long_description_content_type="text/markdown",
    url="",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        'certifi',
        'cycler',
        'decorator',
        'future',
        'imageio',
        'kiwisolver',
        'Mako',
        'Markdown',
        'MarkupSafe',
        'matplotlib',
        'networkx',
        'nibabel',
        'numpy',
        'opencv-python',
        'packaging',
        'pandas',
        'Pillow',
        'pyparsing',
        'python-dateutil',
        'pytz',
        'PyWavelets',
        'scikit-image',
        'scipy',
        'SimpleITK',
        'six',
    ],
)
