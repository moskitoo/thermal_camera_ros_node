from setuptools import setup, find_packages

package_name = 'thermal_2'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(include=[package_name, 'ir_py_thermal']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'numpy', 'cv_bridge', 'sensor_msgs', 'rclpy'],
    zip_safe=True,
    maintainer='moskit',
    maintainer_email='tomek_niedzi@o2.pl',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'talker = thermal_2.node:main',
        ],
    },
)
