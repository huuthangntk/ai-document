from setuptools import setup, find_packages

setup(
    name="codebase-doc",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        'click>=8.0.0',
        'pathlib>=1.0.1',
        'gitignore_parser>=0.1.0',
        'rich>=10.0.0',  # For beautiful terminal output
        'typing-extensions>=4.0.0',
        'psutil>=5.8.0',  # For process management
    ],
    entry_points={
        'console_scripts': [
            'docgen=codebase_doc.cli:main',
        ],
    },
    author="Your Name",
    author_email="your.email@example.com",
    description="A documentation generator for Python projects",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    keywords="documentation, generator, python",
    url="https://github.com/yourusername/codebase-doc",
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Documentation',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    python_requires='>=3.8',
)