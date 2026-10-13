from setuptools import find_packages, setup

package_name = 'py01_topic'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ska',
    maintainer_email='ska@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'demo01_talker_str_py = py01_topic.demo01_talker_str_py:main',
            'demo1_listener_str_py = py01_topic.demo1_listener_str_py:main',
            'stereo_depth_publisher = py01_topic.stereo_depth_publisher:main',
        ],
    },
)
