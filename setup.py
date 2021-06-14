import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

from setuptools.command.install import install
class PostInstallExtrasInstaller(install):
    extras_install_by_default = ['opencv-python']

    @classmethod
    def pip_main(cls, *args, **kwargs):
        def pip_main(*args, **kwargs):
            raise Exception('No pip module found')
        try:
            from pip import main as pip_main
        except ImportError:
            from pip._internal import main as pip_main

        ret = pip_main(*args, **kwargs)
        if ret:
            raise Exception(f'Exitcode {ret}')
        return ret

    def run(self):
        for extra in self.extras_install_by_default:
            try:
                self.pip_main(['install', extra])
            except Exception as E:
                print(f'Optional package {extra} not installed: {E}')
            else:
                print(f"Optional package {extra} installed")
        return install.run(self)

setuptools.setup(
    name="medcam",
    version="0.1.19",
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
    extras_require={
        'extras': PostInstallExtrasInstaller.extras_install_by_default,
    },
    cmdclass={
        'install': PostInstallExtrasInstaller,
    },
)
