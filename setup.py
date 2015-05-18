#! /usr/bin/env python2.5
# -*- coding: utf-8 -*-


__author__ = 'Justin S Bayer, bayer.justin@googlemail.com'


from setuptools import setup, find_packages


setup(
    name="PyBrain2",
    version="0.4.0",
    description="PyBrain2 is the modestly improved PyBrain, the Swiss army knife for neural networking.",
    license="BSD",
    maintainer="Hobson Lane <pybrain2@totalgood.com> and Chris Morgan <chris.j.morgan@gmail.com>",
    maintainer_email="pybrain2@totalgood.com", 
    keywords="Neural Network Machine Learning",
    url="http://github.com/pybrain2/",
    packages=find_packages(exclude=['examples', 'docs']),
    include_package_data=True,
    test_suite='pybrain2.tests.runtests.make_test_suite',
    package_data={'pybrain': ['rl/environments/ode/models/*.xode']},
    install_requires = ["scipy"],
)
