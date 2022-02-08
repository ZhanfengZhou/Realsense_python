from setuptools import setup
import os
from glob import glob

package_name = 'camera_calibration'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ("share/" + package_name, glob("launch/*.launch.py")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='zhanfeng',
    maintainer_email='zhanfeng.zhou@mail.utoronto.ca',
    description='camera l515 calibration node',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "timer_node_image_capture = \
                camera_calibration.timer_node_image_capture:main",
        ],
    },
)
