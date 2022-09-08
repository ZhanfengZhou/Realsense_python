from setuptools import setup

package_name = 'aruco_stuff'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='matthewrhdu',
    maintainer_email='matthewrhdu@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "pub = aruco_stuff.aruco_publisher:main",
            "sub = aruco_stuff.aruco_subscriber:main",
            "pub_xyz = aruco_stuff.aruco_publisher_xyz:main",
            "pub_pose = aruco_stuff.aruco_publisher_pose:main",
        ],
    },
)
