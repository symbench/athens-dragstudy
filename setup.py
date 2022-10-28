# Copyright (C) 2022, Michael Sandborn
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

from setuptools import setup

setup(
    name='athens-dragstudy',
    version='0.1',
    packages=['athens_dragstudy'],
    license='GPL 3',
    description="scripts to generate and analyze UAV/UAM drag data",
    long_description=open('README.md').read(),
    python_requires='>3.6',
    # do not list standard packages
    install_requires=[
        "matplotlib",
        "numpy",
        "creopyson",
        "trimesh",
    ],
    entry_points={
        'console_scripts': [
            'athens-dragstudy = athens_dragstudy.__main__:run'
        ]
    }
)
