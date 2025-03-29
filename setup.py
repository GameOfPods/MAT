#  MAT - Toolkit to analyze media
#  Copyright (c) 2025.  RedRem95
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.


import os

import setuptools

with open(os.path.join(os.path.dirname(__file__), "MAT", "version.txt"), "r", encoding="utf-8") as fv:
    __version__ = fv.read().strip()

with open("README.md", "r", encoding="utf-8") as fd:
    long_description = fd.read()

with open("requirements.txt", "r", encoding="utf-8") as fr:
    requirements = []
    extra_dependency_links = []
    for line in (x.strip() for x in fr.readlines() if len(x.strip()) > 0):
        if line.startswith("#"):
            continue
        if line.startswith("--extra-index-url"):
            extra_dependency_links.append(line[len("--extra-index-url"):].strip())
        else:
            if line.startswith("git+"):
                if "#egg=" in line:
                    egg_name = line.split("#egg=")[-1]
                else:
                    egg_name = line.split("/")[-1].split(".git")[0]
                line = f"{egg_name} @ {line}"
            requirements.append(line)

print(requirements, extra_dependency_links)

setuptools.setup(
    name="MAT",
    version=__version__,
    author="RedRem",
    description="Media Analytics Toolkit",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license_files=('LICENSE',),
    url="https://github.com/GameOfPods/MAT",
    project_urls={
        "Bug Tracker": "https://github.com/GameOfPods/MAT/issues",
    },
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        "License :: OSI Approved :: GPL-3.0-only",
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.12',
    ],
    entry_points={
        'console_scripts': [
            'MAT = MAT.__main__:main'
        ],
    },
    packages=setuptools.find_packages(include=['MAT', 'MAT.*']),
    python_requires=">=3.9",
    install_requires=requirements,
    dependency_links=extra_dependency_links + ['https://pypi.example.org/pypi/somedep/'],
    include_package_data=True,
    zip_safe=False,
)
