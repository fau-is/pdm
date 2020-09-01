# coding=utf-8
"""Contains methods for command line interaction"""
import argparse


def parse_args():
    """
    Creates the argument parser for the command line
    :return: the args that were parsed from the command line
    """
    # This file contains the commandline tools for the current tool
    parser = argparse.ArgumentParser(prog='main.py', usage='main.py [options]')

    parser.add_argument('--eventLog', nargs='?', default='Resources/a_log17.xes',
                        help='The path pulling the event log')

    parser.add_argument('--XmlDcr', nargs="?", default='Resources/BPI_A2.xml',
                        help='The input path for the DCR Graph xml')

    parser.add_argument('--outputPathPCM', nargs="?", default="./",
                        help='The output path for a csv event log')

    return parser.parse_args()
