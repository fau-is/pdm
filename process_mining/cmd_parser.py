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

    parser.add_argument('--eventLog', nargs='?', default='Resources/Hospital Billing - Event Log.xes',
                        help='The path pulling the event log')

    parser.add_argument('--XmlDcr', nargs="?", default='Resources/Hospital Billing.xml',
                        help='The input path for the DCR Graph xml')

    parser.add_argument('--outputPathPCM', nargs="?", default="./Output/train_hb_pcm_2.csv",
                        help='The output path for a csv event log')

    parser.add_argument('--outputPathNEP', nargs="?", default="./Output/train_hb_nep.csv",
                        help='The output path for a csv event log')

    return parser.parse_args()
